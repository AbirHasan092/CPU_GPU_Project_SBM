/**
 * SBM — Simulated Bifurcation Machine (MaxCut)
 * Runs on either CPU or GPU — selected at runtime via --device cpu|gpu
 *
 * GPU path (default): CUDA-accelerated, FP32, optimised for RTX 2060 (sm_75)
 *   • FP32 throughout GPU kernels — Turing FP64 runs at 1/32 of FP32
 *   • bSB and dSB on separate CUDA streams via std::thread (concurrent)
 *   • __ldg() on read-only CSR arrays → L1 texture-cache path
 *
 * CPU path (--device cpu): pure C++, FP64, multithreaded bSB+dSB
 *   • Uses all available FP64 throughput on the host
 *   • bSB and dSB run concurrently on two std::threads
 *   • No CUDA required
 *
 * Compile:
 *   nvcc -O3 -arch=sm_75 -use_fast_math -o SBM SBM_CPU.cpp -lcurand
 *   (Rename to SBM_CPU.cu if nvcc rejects the .cpp extension.)
 *
 * Usage:
 *   ./SBM [--device cpu|gpu] [--flag value ...] [file1.txt file2.txt ...]
 *
 * Requires: NVIDIA RTX 2060 (or any sm_75+ card) + CUDA Toolkit >= 11.0
 *
 * Reference:
 *   Goto et al., "High-performance combinatorial optimization based on
 *   classical mechanics", Science Advances 7(6), 2021.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <thread>
using namespace std;

// ─── CUDA error-check macro ──────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── Utilities ───────────────────────────────────────────────────────────────
static void stripCR(string& s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
}
static bool isInt(const string& s) {
    if (s.empty()) return false;
    size_t i = (s[0] == '-') ? 1 : 0;
    for (; i < s.size(); ++i) if (!isdigit((unsigned char)s[i])) return false;
    return true;
}

// ─── Graph (host) ────────────────────────────────────────────────────────────
struct Graph {
    string name;
    int    n = 0, m = 0;
    double p = 0.0;
    vector<vector<int>> adj;
};

static const char* NOISE_LABEL[] = { "White noise", "Spin noise  ", "Coupling noise" };
static const char* NOISE_SHORT[] = { "White",       "Spin",         "Coupling"       };

struct Result {
    string name;
    int    n, m;
    int    bestCut;
    int    minCut, maxCut;
    double meanCut, stdCut;
    double totalMs;
    int    trials;
    int    noiseMode;  // 0=white  1=spin (Kn*xi*dW)  2=coupling (Kn*(xi+xj)*dW)
};

bool loadGraph(const string& filename, Graph& g) {
    ifstream f(filename);
    if (!f.is_open()) return false;

    bool dimacs = false, has_n = false, isMTX = false;
    vector<pair<int,int>> raw;
    string line;

    while (getline(f, line)) {
        stripCR(line);
        if (line.empty() || line[0] == '#') continue;
        if (line[0] == 'c' && (line.size()==1 || isspace((unsigned char)line[1]))) continue;
        if (line[0] == '%') {
            if (line.find("MatrixMarket") != string::npos) isMTX = true;
            continue;
        }

        istringstream ss(line); string tok; ss >> tok;
        if (tok == "n")           { ss >> g.n; has_n = true; continue; }
        if (tok == "p" && !has_n) { string t2; ss >> t2;
                                    if (t2 == "edge") { ss >> g.n >> g.m; has_n = dimacs = true; }
                                    continue; }
        if (tok == "p" && has_n)  { continue; }
        if (tok == "m")           { ss >> g.m; continue; }
        if (dimacs && (tok=="e"||tok=="E")) {
            int u, v; ss >> u >> v; raw.push_back({u-1, v-1}); continue;
        }
        // MatrixMarket size line: "rows cols nnz" — nodes = rows, skip cols (same for square)
        if (isMTX && !has_n && isInt(tok)) {
            int cols, nnz; g.n = stoi(tok); ss >> cols >> nnz; g.m = nnz; has_n = true; continue;
        }
        if (!has_n && isInt(tok)) { g.n = stoi(tok); ss >> g.m; has_n = true; continue; }
        if (isInt(tok)) {
            int u = stoi(tok), v;
            if (ss >> v) {
                if (isMTX) raw.push_back({u - 1, v - 1});  // MTX is 1-indexed
                else        raw.push_back({u, v});
            }
        }
    }

    if (!has_n) {
        int mx = -1;
        for (auto& e : raw) mx = max(mx, max(e.first, e.second));
        if (mx < 0) return false;
        g.n = mx + 1;
    }
    if (g.n <= 0) return false;

    g.adj.assign(g.n, {});
    for (auto& e : raw) {
        int u = e.first, v = e.second;
        if (u < 0 || v < 0 || u >= g.n || v >= g.n || u == v) continue;
        g.adj[u].push_back(v);
        g.adj[v].push_back(u);
    }
    int s = 0;
    for (int i = 0; i < g.n; ++i) s += (int)g.adj[i].size();
    g.m = s / 2;
    return g.n > 0 && g.m > 0;
}

// ─── CSR graph on device ─────────────────────────────────────────────────────
struct GPUGraph {
    int  n = 0, m = 0;
    int* d_rowPtr = nullptr;
    int* d_colIdx = nullptr;

    void upload(const Graph& g) {
        n = g.n; m = g.m;
        int nnz = 2 * m;
        vector<int> rowPtr(n + 1, 0);
        vector<int> colIdx;
        colIdx.reserve(nnz);
        for (int i = 0; i < n; ++i) rowPtr[i+1] = rowPtr[i] + (int)g.adj[i].size();
        for (int i = 0; i < n; ++i) for (int j : g.adj[i]) colIdx.push_back(j);
        CUDA_CHECK(cudaMalloc(&d_rowPtr, (n+1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_colIdx,  nnz   * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(), (n+1)*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx.data(),  nnz *sizeof(int), cudaMemcpyHostToDevice));
    }

    void free_() {
        cudaFree(d_rowPtr); cudaFree(d_colIdx);
        d_rowPtr = nullptr; d_colIdx = nullptr;
    }
};

// ─── SBM parameters ──────────────────────────────────────────────────────────
struct SBMParams {
    double dt       = 0.10;
    int    steps    = -1;
    double a0       = 1.0;
    double Delta    = 1.0;
    double c        = 0.5;
    double xi       = -1.0;
    double noise    = 0.02;
    int    restarts = -1;
    int    trials   = 5;
    bool   postLS   = true;
    bool   useCPU   = true;   // --device cpu → true, --device gpu → false
};

SBMParams scaleParams(const SBMParams& base, int n) {
    SBMParams p = base;
    if (p.steps   < 0) p.steps   = max(1000, (int)(1000.0 * pow((double)n / 50.0, 0.6)));
    if (p.restarts < 0) p.restarts = max(5,  (int)(20.0   * pow(50.0 / max(n, 50), 0.5)));
    return p;
}

// ─── MaxCut eval + greedy local search (CPU) ─────────────────────────────────
int evalCut(const Graph& g, const vector<int>& spins) {
    int cut = 0;
    for (int u = 0; u < g.n; ++u)
        for (int v : g.adj[u])
            if (v > u && spins[u] != spins[v]) ++cut;
    return cut;
}

void localSearch(const Graph& g, vector<int>& spins) {
    bool imp = true;
    while (imp) {
        imp = false;
        for (int i = 0; i < g.n; ++i) {
            int gain = 0;
            for (int j : g.adj[i]) gain += (spins[i] == spins[j]) ? 1 : -1;
            if (gain > 0) { spins[i] = -spins[i]; imp = true; }
        }
    }
}

// ─── Shared trial-result printer ─────────────────────────────────────────────
static void printGraphHeader(const Graph& g, const SBMParams& p, int noiseMode) {
    string info = "nodes=" + to_string(g.n) + "  edges=" + to_string(g.m);
    if (g.p > 0) info += "  p=" + to_string(g.p).substr(0, 4);
    string cfg = "steps=" + to_string(p.steps) + "  restarts=" + to_string(p.restarts)
               + "  trials=" + to_string(p.trials);
    string noiseLbl = string("noise=") + NOISE_LABEL[noiseMode];
    cout << "\n+----------------------------------------------------+\n";
    cout << "|  " << left << setw(50) << g.name    << "|\n";
    cout << "|  " << left << setw(50) << info      << "|\n";
    cout << "|  " << left << setw(50) << cfg       << "|\n";
    cout << "|  " << left << setw(50) << noiseLbl  << "|\n";
    cout << "+----------------------------------------------------+\n";
    cout << fixed << setprecision(2);
    cout << "  Trial  |  bSB cut  |  dSB cut  |  Best  |  Time (ms)\n";
    cout << "  -------+-----------+-----------+--------+-----------\n";
}

static Result printGraphFooter(const Graph& g, const vector<int>& trialBests,
                                const vector<int>& bestSpins, double totalTime,
                                int trials, int noiseMode) {
    int    mn   = *min_element(trialBests.begin(), trialBests.end());
    int    mx   = *max_element(trialBests.begin(), trialBests.end());
    double mean = accumulate(trialBests.begin(), trialBests.end(), 0.0) / trials;
    double sq   = 0.0; for (int c : trialBests) sq += (c - mean) * (c - mean);
    double stdv = sqrt(sq / trials);
    double ratio = (g.m > 0) ? 100.0 * mx / g.m : 0.0;

    cout << "  -------+-----------+-----------+--------+-----------\n";
    cout << "  min="  << mn << "  max=" << mx << "  mean=" << mean
         << "  std="  << setprecision(1) << stdv << "\n";
    cout << setprecision(2);
    cout << "  Best cut = " << mx << "  (" << ratio << "% of all edges)\n";
    cout << "  Total time = " << totalTime * 1000 << " ms"
         << "  (avg " << totalTime * 1000 / trials << " ms/trial)\n";

    if (g.n <= 100) {
        cout << "  Best partition:\n    S0(+1): {";
        bool first = true;
        for (int i = 0; i < g.n; ++i)
            if (bestSpins[i] == +1) { if (!first) cout << ","; cout << i; first = false; }
        cout << "}\n    S1(-1): {"; first = true;
        for (int i = 0; i < g.n; ++i)
            if (bestSpins[i] == -1) { if (!first) cout << ","; cout << i; first = false; }
        cout << "}\n";
    } else {
        int s0 = (int)count(bestSpins.begin(), bestSpins.end(), +1);
        cout << "  Best partition:  S0(+1): " << s0
             << " nodes  |  S1(-1): " << (g.n - s0) << " nodes\n";
    }

    return { g.name, g.n, g.m, mx, mn, mx, mean, stdv, totalTime * 1000, trials, noiseMode };
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CPU SOLVER
// ═══════════════════════════════════════════════════════════════════════════════

// One variant of the SBM algorithm in pure C++ (FP64, no CUDA).
//   mode      = 0: bSB (ballistic)       — coupling uses x[j]
//   mode      = 1: dSB (discrete)        — coupling uses sign(x[j])
//   noiseMode = 0: white noise           — nf * N(0,1)
//   noiseMode = 1: spin-dependent        — nf * x[i] * N(0,1)
//   noiseMode = 2: coupling-dependent    — nf * (x[i] + mean_j(x[j])) * N(0,1)
vector<int> solveCPU(const Graph& g, const SBMParams& p, int mode, int noiseMode,
                     unsigned long long seed)
{
    int n = g.n;

    double xi = p.xi;
    if (xi < 0.0) {
        int mx = 0;
        for (int i = 0; i < n; ++i) mx = max(mx, (int)g.adj[i].size());
        xi = (mx > 0) ? 0.7 / mx : 0.7;
    }
    double nf = p.noise * sqrt(p.dt);

    mt19937_64 rng(seed);
    uniform_real_distribution<double> udist(-0.02, 0.02);
    normal_distribution<double>       ndist(0.0, 1.0);

    vector<double> x(n), y(n), h(n);
    vector<int>    bestSpins(n, 1), h_spins(n);
    int            bestCut = -1;

    for (int run = 0; run < p.restarts; ++run) {
        // Random initialisation
        for (int i = 0; i < n; ++i) { x[i] = udist(rng); y[i] = udist(rng); }

        // Time-stepping loop
        for (int t = 0; t < p.steps; ++t) {
            double pump = p.a0 * (double)t / p.steps;   // linear ramp 0 → a0

            // Coupling field
            for (int i = 0; i < n; ++i) {
                double s = 0.0;
                if (mode == 0)
                    for (int j : g.adj[i]) s += x[j];
                else
                    for (int j : g.adj[i]) s += (x[j] >= 0.0) ? 1.0 : -1.0;
                h[i] = s;
            }

            // Momentum Euler step + noise (mode-dependent)
            for (int i = 0; i < n; ++i) {
                double xi3 = x[i] * x[i] * x[i];
                y[i] += p.dt * (-(p.a0 - pump)*x[i] - p.a0*p.c*xi3 + xi*h[i]);
                if (nf > 0.0) {
                    double dW = ndist(rng);
                    if      (noiseMode == 1) { dW *= x[i]; }
                    else if (noiseMode == 2) {
                        int deg = (int)g.adj[i].size();
                        dW *= x[i] + (deg > 0 ? h[i] / deg : 0.0);
                    }
                    y[i] += nf * dW;
                }
            }

            // Position Euler step + wall clamp
            for (int i = 0; i < n; ++i) {
                x[i] += p.dt * p.Delta * y[i];
                if (x[i] >  1.0) { x[i] =  1.0; y[i] = 0.0; }
                if (x[i] < -1.0) { x[i] = -1.0; y[i] = 0.0; }
            }
        }

        // Binarize → optional local search → track best
        for (int i = 0; i < n; ++i) h_spins[i] = (x[i] >= 0.0) ? 1 : -1;
        if (p.postLS) localSearch(g, h_spins);
        int cut = evalCut(g, h_spins);
        if (cut > bestCut) { bestCut = cut; bestSpins = h_spins; }
    }
    return bestSpins;
}

// Per-graph driver for the CPU path.
// bSB and dSB are launched on two std::threads (concurrent host execution).
Result solveGraphCPU(const Graph& g, const SBMParams& base, int noiseMode) {
    SBMParams p = scaleParams(base, g.n);
    printGraphHeader(g, p, noiseMode);

    mt19937_64  seedGen(random_device{}());
    vector<int> bestSpins;
    int         overallBest = -1;
    double      totalTime   = 0.0;
    vector<int> trialBests(p.trials);

    for (int tr = 0; tr < p.trials; ++tr) {
        unsigned long long seed = seedGen();
        auto t0 = chrono::high_resolution_clock::now();

        // Run bSB and dSB concurrently on two CPU threads
        vector<int> sb, sd;
        thread t_bsb([&]() { sb = solveCPU(g, p, 0, noiseMode, seed); });
        thread t_dsb([&]() { sd = solveCPU(g, p, 1, noiseMode, seed ^ 0xDEADBEEFULL); });
        t_bsb.join();
        t_dsb.join();

        double elapsed = chrono::duration<double>(
                             chrono::high_resolution_clock::now() - t0).count();
        totalTime += elapsed;

        int cb = evalCut(g, sb), cd = evalCut(g, sd), best = max(cb, cd);
        trialBests[tr] = best;

        cout << "    " << setw(3) << (tr+1) << "    |"
             << "   " << setw(6) << cb      << "    |"
             << "   " << setw(6) << cd      << "    |"
             << "  "  << setw(6) << best    << "  |"
             << "  "  << fixed << setprecision(2) << elapsed * 1000 << "\n";

        if (best > overallBest) { overallBest = best; bestSpins = (cb >= cd) ? sb : sd; }
    }

    return printGraphFooter(g, trialBests, bestSpins, totalTime, p.trials, noiseMode);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  GPU SOLVER  (FP32 kernels — tuned for RTX 2060 / Turing sm_75)
// ═══════════════════════════════════════════════════════════════════════════════

static const int BLOCK = 256;

__global__ void k_initRNG(curandState* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, (unsigned long long)i, 0, &states[i]);
}

__global__ void k_initOsc(float* x, float* y, curandState* states, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = 0.04f * curand_uniform(&states[i]) - 0.02f;
    y[i] = 0.04f * curand_uniform(&states[i]) - 0.02f;
}

// bSB coupling  h[i] = sum_{j in N(i)} x[j]
// __ldg routes CSR index reads through the L1 texture cache.
__global__ void k_couplingBSB(const float* __restrict__ x,
                               const int*   __restrict__ rowPtr,
                               const int*   __restrict__ colIdx,
                               float* h, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    int start = __ldg(&rowPtr[i]), end = __ldg(&rowPtr[i + 1]);
    for (int k = start; k < end; ++k) { int nb = __ldg(&colIdx[k]); s += __ldg(&x[nb]); }
    h[i] = s;
}

// dSB coupling  h[i] = sum_{j in N(i)} sign(x[j])
__global__ void k_couplingDSB(const float* __restrict__ x,
                               const int*   __restrict__ rowPtr,
                               const int*   __restrict__ colIdx,
                               float* h, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    int start = __ldg(&rowPtr[i]), end = __ldg(&rowPtr[i + 1]);
    for (int k = start; k < end; ++k) { int nb = __ldg(&colIdx[k]); s += (__ldg(&x[nb]) >= 0.0f) ? 1.0f : -1.0f; }
    h[i] = s;
}

// Momentum Euler step — white noise:  nf * N(0,1)
__global__ void k_stepY(float* y, const float* x, const float* h,
                         float pump, float a0, float c, float xi,
                         float dt,   float nf,
                         curandState* states, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi3 = x[i] * x[i] * x[i];
    y[i] += dt * (-(a0 - pump) * x[i]  -  a0 * c * xi3  +  xi * h[i]);
    if (nf > 0.0f) y[i] += nf * curand_normal(&states[i]);
}

// Momentum Euler step — spin-dependent noise:  nf * x[i] * N(0,1)
// Oscillators near ±1 (strongly committed) receive proportionally more noise.
__global__ void k_stepY_spin(float* y, const float* x, const float* h,
                              float pump, float a0, float c, float xi,
                              float dt,   float nf,
                              curandState* states, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi3 = x[i] * x[i] * x[i];
    y[i] += dt * (-(a0 - pump) * x[i]  -  a0 * c * xi3  +  xi * h[i]);
    if (nf > 0.0f) y[i] += nf * x[i] * curand_normal(&states[i]);
}

// Momentum Euler step — coupling-dependent noise:  nf * (x[i] + mean_j(x[j])) * N(0,1)
// Scale = x[i] + h[i]/deg[i].  In-phase pairs (same sign) get scale~2, anti-phase get ~0.
// Naturally destabilises ferromagnetic clusters while leaving MaxCut edges quiet.
__global__ void k_stepY_coup(float* y, const float* x, const float* h,
                              const int* __restrict__ rowPtr,
                              float pump, float a0, float c, float xi,
                              float dt,   float nf,
                              curandState* states, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi3 = x[i] * x[i] * x[i];
    y[i] += dt * (-(a0 - pump) * x[i]  -  a0 * c * xi3  +  xi * h[i]);
    if (nf > 0.0f) {
        int deg = __ldg(&rowPtr[i + 1]) - __ldg(&rowPtr[i]);
        float scale = x[i] + (deg > 0 ? h[i] / (float)deg : 0.0f);
        y[i] += nf * scale * curand_normal(&states[i]);
    }
}

// Position Euler step + wall clamp
__global__ void k_stepX(float* x, float* y, float dt, float delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += dt * delta * y[i];
    if (x[i] >  1.0f) { x[i] =  1.0f; y[i] = 0.0f; }
    if (x[i] < -1.0f) { x[i] = -1.0f; y[i] = 0.0f; }
}

__global__ void k_binarize(const float* x, int* spins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) spins[i] = (x[i] >= 0.0f) ? 1 : -1;
}

// One variant of the SBM algorithm on the GPU (FP32, given CUDA stream).
vector<int> solveGPU(const Graph& g, const GPUGraph& gg,
                      const SBMParams& p, int mode, int noiseMode,
                      unsigned long long seed, cudaStream_t stream)
{
    int n      = g.n;
    int blocks = (n + BLOCK - 1) / BLOCK;

    double xi_d = p.xi;
    if (xi_d < 0.0) {
        int mx = 0;
        for (int i = 0; i < n; ++i) mx = max(mx, (int)g.adj[i].size());
        xi_d = (mx > 0) ? 0.7 / mx : 0.7;
    }

    const float fdt   = (float)p.dt;
    const float fa0   = (float)p.a0;
    const float fc    = (float)p.c;
    const float fxi   = (float)xi_d;
    const float fDelt = (float)p.Delta;
    const float fnf   = (float)(p.noise * sqrt(p.dt));

    float*       d_x;
    float*       d_y;
    float*       d_h;
    int*         d_spins;
    curandState* d_states;

    CUDA_CHECK(cudaMalloc(&d_x,      n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,      n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h,      n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spins,  n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));

    k_initRNG<<<blocks, BLOCK, 0, stream>>>(d_states, seed, n);

    vector<int> bestSpins(n, 1);
    int         bestCut = -1;
    vector<int> h_spins(n);

    for (int run = 0; run < p.restarts; ++run) {
        k_initOsc<<<blocks, BLOCK, 0, stream>>>(d_x, d_y, d_states, n);

        for (int t = 0; t < p.steps; ++t) {
            float pump = fa0 * ((float)t / p.steps);

            if (mode == 0)
                k_couplingBSB<<<blocks, BLOCK, 0, stream>>>(d_x, gg.d_rowPtr, gg.d_colIdx, d_h, n);
            else
                k_couplingDSB<<<blocks, BLOCK, 0, stream>>>(d_x, gg.d_rowPtr, gg.d_colIdx, d_h, n);

            if (noiseMode == 1)
                k_stepY_spin<<<blocks, BLOCK, 0, stream>>>(d_y, d_x, d_h,
                                                            pump, fa0, fc, fxi,
                                                            fdt, fnf, d_states, n);
            else if (noiseMode == 2)
                k_stepY_coup<<<blocks, BLOCK, 0, stream>>>(d_y, d_x, d_h, gg.d_rowPtr,
                                                            pump, fa0, fc, fxi,
                                                            fdt, fnf, d_states, n);
            else
                k_stepY<<<blocks, BLOCK, 0, stream>>>(d_y, d_x, d_h,
                                                       pump, fa0, fc, fxi,
                                                       fdt, fnf, d_states, n);
            k_stepX<<<blocks, BLOCK, 0, stream>>>(d_x, d_y, fdt, fDelt, n);
        }

        k_binarize<<<blocks, BLOCK, 0, stream>>>(d_x, d_spins, n);
        CUDA_CHECK(cudaMemcpyAsync(h_spins.data(), d_spins,
                                   n * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (p.postLS) localSearch(g, h_spins);
        int cut = evalCut(g, h_spins);
        if (cut > bestCut) { bestCut = cut; bestSpins = h_spins; }
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_h);
    cudaFree(d_spins); cudaFree(d_states);
    return bestSpins;
}

// Per-graph driver for the GPU path.
// bSB and dSB are launched on separate CUDA streams via two std::threads.
Result solveGraphGPU(const Graph& g, const GPUGraph& gg, const SBMParams& base, int noiseMode) {
    SBMParams p = scaleParams(base, g.n);
    printGraphHeader(g, p, noiseMode);

    cudaStream_t stream0, stream1;
    CUDA_CHECK(cudaStreamCreate(&stream0));
    CUDA_CHECK(cudaStreamCreate(&stream1));

    mt19937_64  seedGen(random_device{}());
    vector<int> bestSpins;
    int         overallBest = -1;
    double      totalTime   = 0.0;
    vector<int> trialBests(p.trials);

    for (int tr = 0; tr < p.trials; ++tr) {
        unsigned long long seed = seedGen();
        auto t0 = chrono::high_resolution_clock::now();

        vector<int> sb, sd;
        thread t_bsb([&]() { sb = solveGPU(g, gg, p, 0, noiseMode, seed,                 stream0); });
        thread t_dsb([&]() { sd = solveGPU(g, gg, p, 1, noiseMode, seed ^ 0xDEADBEEFULL, stream1); });
        t_bsb.join();
        t_dsb.join();

        double elapsed = chrono::duration<double>(
                             chrono::high_resolution_clock::now() - t0).count();
        totalTime += elapsed;

        int cb = evalCut(g, sb), cd = evalCut(g, sd), best = max(cb, cd);
        trialBests[tr] = best;

        cout << "    " << setw(3) << (tr+1) << "    |"
             << "   " << setw(6) << cb      << "    |"
             << "   " << setw(6) << cd      << "    |"
             << "  "  << setw(6) << best    << "  |"
             << "  "  << fixed << setprecision(2) << elapsed * 1000 << "\n";

        if (best > overallBest) { overallBest = best; bestSpins = (cb >= cd) ? sb : sd; }
    }

    CUDA_CHECK(cudaStreamDestroy(stream0));
    CUDA_CHECK(cudaStreamDestroy(stream1));

    return printGraphFooter(g, trialBests, bestSpins, totalTime, p.trials, noiseMode);
}

// ─── CLI argument parser ──────────────────────────────────────────────────────
static void printHelp(const char* prog) {
    cout <<
"Usage:  " << prog << " [--device cpu|gpu] [--flag value ...] [file1.txt ...]\n"
"\nDevice:\n"
"  --device   cpu|gpu   Execution target                  default: gpu\n"
"                         gpu  — CUDA FP32 kernels on RTX 2060 (sm_75)\n"
"                         cpu  — pure C++ FP64 on the host CPU\n"
"\nEuler integration:\n"
"  --dt       <float>   Step size (convergence rate)      default: 0.10\n"
"  --steps    <int>     Time steps per restart            default: auto\n"
"\nSBM physics:\n"
"  --a0       <float>   Pump amplitude                    default: 1.0\n"
"  --c        <float>   Kerr nonlinear coefficient        default: 0.5\n"
"  --xi       <float>   Coupling strength (-1=auto)       default: auto\n"
"  --delta    <float>   Frequency parameter               default: 1.0\n"
"\nNoise (stochastic SB):\n"
"  --noise    <float>   Gaussian noise std per step       default: 0.02\n"
"                       Injected as sigma*sqrt(dt)*N(0,1) into momentum.\n"
"\nSearch:\n"
"  --restarts <int>     Restarts per trial                default: auto\n"
"  --trials   <int>     Independent trials per graph      default: 5\n"
"  --no-ls              Disable greedy 1-flip local search\n"
"\nFiles:\n"
"  Positional args are graph file paths (.mtx MatrixMarket or DIMACS .txt).\n"
"  Default (no files given): all GSET\\G*.mtx files.\n"
"\nExamples:\n"
"  " << prog << " --device gpu GSET\\G1.mtx\n"
"  " << prog << " --device cpu --trials 3 GSET\\G1.mtx GSET\\G2.mtx\n"
"  " << prog << " --device gpu --steps 3000 --noise 0.01 GSET\\G14.mtx\n";
}

SBMParams parseArgs(int argc, char* argv[], vector<string>& files) {
    SBMParams p;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--help" || a == "-h") { printHelp(argv[0]); exit(0); }
        if (a == "--no-ls") { p.postLS = false; continue; }

        auto need = [&]() -> const char* {
            if (i + 1 >= argc) { cerr << "[ERROR] " << a << " requires a value.\n"; exit(1); }
            return argv[++i];
        };

        if      (a == "--device") {
            string d = need();
            if      (d == "cpu") p.useCPU = true;
            else if (d == "gpu") p.useCPU = false;
            else cerr << "[WARN] Unknown device '" << d << "' — defaulting to gpu.\n";
        }
        else if (a == "--dt")       p.dt       = stod(need());
        else if (a == "--steps")    p.steps    = stoi(need());
        else if (a == "--restarts") p.restarts = stoi(need());
        else if (a == "--trials")   p.trials   = stoi(need());
        else if (a == "--a0")       p.a0       = stod(need());
        else if (a == "--c")        p.c        = stod(need());
        else if (a == "--xi")       p.xi       = stod(need());
        else if (a == "--delta")    p.Delta    = stod(need());
        else if (a == "--noise")    p.noise    = stod(need());
        else if (a[0] == '-')       cerr << "[WARN] Unknown flag: " << a << "\n";
        else                        files.push_back(a);
    }
    return p;
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    vector<string> files;
    SBMParams params = parseArgs(argc, argv, files);

    // ── Print banner (device-dependent) ──────────────────────────────────────
    cout << "======================================================\n";
    cout << "  SBM — Simulated Bifurcation Machine (MaxCut)\n";
    if (params.useCPU) {
        cout << "  Device : CPU (FP64, two std::threads per trial)\n";
        cout << "  Variants: bSB (ballistic)  |  dSB (discrete)\n";
    } else {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        cout << "  Device : GPU — " << prop.name
             << "  (" << prop.multiProcessorCount << " SMs, "
             << prop.totalGlobalMem / 1024 / 1024 << " MB VRAM)\n";
        cout << "  Precision: FP32 kernels (optimal for Turing / RTX 20xx)\n";
        cout << "  Variants: bSB (ballistic)  |  dSB (discrete)  [concurrent streams]\n";
    }
    cout << "  Reference: Goto et al., Science Advances 7(6), 2021\n";
    cout << "======================================================\n";

    // ── Default file glob ─────────────────────────────────────────────────────
    if (files.empty()) {
        FILE* ls = _popen("dir /b GSET\\G*.mtx 2>nul", "r");
        if (ls) {
            char buf[512];
            while (fgets(buf, sizeof(buf), ls)) {
                string s(buf); stripCR(s);
                if (!s.empty() && s.back() == '\n') s.pop_back();
                stripCR(s);
                if (!s.empty()) files.push_back("GSET\\" + s);
            }
            _pclose(ls);
        }
        if (files.empty()) {
            cerr << "[ERROR] No GSET\\G*.mtx files found. "
                    "Pass files explicitly or use --help.\n";
            return 1;
        }
        // Sort numerically by graph number (G1.mtx < G2.mtx < ... < G10.mtx)
        sort(files.begin(), files.end(), [](const string& a, const string& b) {
            auto numOf = [](const string& p) {
                size_t s = p.rfind('G'), e = p.rfind('.');
                if (s == string::npos || e == string::npos || e <= s + 1) return 0;
                try { return stoi(p.substr(s + 1, e - s - 1)); } catch (...) { return 0; }
            };
            return numOf(a) < numOf(b);
        });
        cout << "\nDefaulting to " << files.size()
             << " file(s) in GSET\\G*.mtx\n";
    }

    // ── Print active parameter set ────────────────────────────────────────────
    SBMParams ref = scaleParams(params, 300);
    cout << "\nActive parameters:\n";
    cout << "  Device :  " << (params.useCPU ? "cpu (FP64)" : "gpu (FP32)") << "\n";
    cout << "  Euler  :  dt=" << params.dt
         << "  steps="
         << (params.steps   < 0 ? to_string(ref.steps)   + "(auto)" : to_string(params.steps))
         << "  restarts="
         << (params.restarts < 0 ? to_string(ref.restarts) + "(auto)" : to_string(params.restarts)) << "\n";
    cout << "  Physics:  a0=" << params.a0 << "  c=" << params.c
         << "  xi=" << (params.xi < 0 ? "auto" : to_string(params.xi))
         << "  delta=" << params.Delta << "\n";
    cout << "  Noise  :  sigma=" << params.noise
         << (params.noise == 0.0 ? "  (deterministic)" : "  (stochastic SB)") << "\n";
    cout << "  Search :  trials=" << params.trials
         << "  post-LS=" << (params.postLS ? "on" : "off") << "\n";

    // ── Solve each graph across all 3 noise modes ─────────────────────────────
    int solved = 0, failed = 0;
    // results[graph_idx][noise_mode 0..2]
    vector<array<Result, 3>> allResults;

    for (auto& fname : files) {
        Graph g;
        size_t sl   = fname.find_last_of("/\\");
        string base = (sl == string::npos) ? fname : fname.substr(sl + 1);
        size_t dot  = base.rfind('.');
        g.name      = (dot == string::npos) ? base : base.substr(0, dot);

        if (!loadGraph(fname, g)) {
            cerr << "\n[ERROR] Cannot open '" << fname << "'.\n";
            ++failed; continue;
        }

        array<Result, 3> row;
        if (params.useCPU) {
            for (int nm = 0; nm < 3; ++nm)
                row[nm] = solveGraphCPU(g, params, nm);
        } else {
            GPUGraph gg; gg.upload(g);
            for (int nm = 0; nm < 3; ++nm)
                row[nm] = solveGraphGPU(g, gg, params, nm);
            gg.free_();
        }
        allResults.push_back(row);
        ++solved;
    }

    if (allResults.empty()) {
        cout << "\n======================================================\n";
        cout << "  Done. Solved 0 graph(s)";
        if (failed) cout << ", failed to load " << failed;
        cout << ".\n";
        return (failed > 0) ? 1 : 0;
    }

    // ── Per-noise-mode summary tables ─────────────────────────────────────────
    size_t nameW = 5;
    for (auto& row : allResults) nameW = max(nameW, row[0].name.size());
    nameW += 1;

    auto printSummaryTable = [&](int nm) {
        cout << "\n";
        cout << "+==========================================================================+\n";
        cout << "|  MAXCUT SUMMARY — " << left << setw(53) << NOISE_LABEL[nm] << "|\n";
        cout << "+==========================================================================+\n";
        cout << "  " << left  << setw((int)nameW) << "Graph"
             << "  " << right << setw(6)  << "Nodes"
             << "  " << right << setw(7)  << "Edges"
             << "  " << right << setw(8)  << "BestCut"
             << "  " << right << setw(7)  << "% Edges"
             << "  " << right << setw(8)  << "Mean"
             << "  " << right << setw(6)  << "Std"
             << "  " << right << setw(10) << "Time(ms)"
             << "\n";
        string sep = "  " + string(nameW,'-') + "  " + string(6,'-') + "  " + string(7,'-')
                   + "  " + string(8,'-') + "  " + string(7,'-') + "  " + string(8,'-')
                   + "  " + string(6,'-') + "  " + string(10,'-');
        cout << sep << "\n";
        double grandTotal = 0.0;
        for (auto& row : allResults) {
            const Result& r = row[nm];
            double pct = (r.m > 0) ? 100.0 * r.bestCut / r.m : 0.0;
            grandTotal += r.totalMs;
            cout << fixed
                 << "  " << left  << setw((int)nameW) << r.name
                 << "  " << right << setw(6)  << r.n
                 << "  " << right << setw(7)  << r.m
                 << "  " << right << setw(8)  << r.bestCut
                 << "  " << right << setw(6)  << setprecision(2) << pct << "%"
                 << "  " << right << setw(8)  << setprecision(1) << r.meanCut
                 << "  " << right << setw(6)  << setprecision(1) << r.stdCut
                 << "  " << right << setw(10) << setprecision(0) << r.totalMs
                 << "\n";
        }
        cout << sep << "\n";
        cout << "  Total wall time: " << fixed << setprecision(0) << grandTotal
             << " ms  |  Graphs: " << solved << "\n";
    };

    for (int nm = 0; nm < 3; ++nm) printSummaryTable(nm);

    // ── Noise comparison table ────────────────────────────────────────────────
    cout << "\n";
    cout << "+==========================================================================+\n";
    cout << "|                     NOISE MODE COMPARISON TABLE                         |\n";
    cout << "+==========================================================================+\n";
    // header: Graph | White BestCut % | Spin BestCut % | Coupling BestCut % | Best Mode
    cout << "  " << left << setw((int)nameW) << "Graph"
         << "  " << right << setw(8) << "White"
         << " " << setw(6) << "%"
         << "  " << right << setw(8) << "Spin"
         << " " << setw(6) << "%"
         << "  " << right << setw(8) << "Coupling"
         << " " << setw(6) << "%"
         << "  " << left << "BestMode"
         << "\n";
    string cmpSep = "  " + string(nameW,'-') + "  " + string(15,'-')
                  + "  " + string(15,'-') + "  " + string(15,'-') + "  " + string(10,'-');
    cout << cmpSep << "\n";

    int winCount[3] = {0, 0, 0};
    for (auto& row : allResults) {
        int cuts[3] = { row[0].bestCut, row[1].bestCut, row[2].bestCut };
        int bestCut = *max_element(cuts, cuts + 3);
        int bestNM  = (int)(max_element(cuts, cuts + 3) - cuts);
        winCount[bestNM]++;
        cout << fixed
             << "  " << left << setw((int)nameW) << row[0].name;
        for (int nm = 0; nm < 3; ++nm) {
            double pct = (row[nm].m > 0) ? 100.0 * cuts[nm] / row[nm].m : 0.0;
            cout << "  " << right << setw(8) << cuts[nm]
                 << " " << setw(5) << setprecision(2) << pct << "%";
        }
        cout << "  " << left << NOISE_SHORT[bestNM]
             << (cuts[bestNM] == bestCut && cuts[0]==cuts[1] && cuts[1]==cuts[2] ? " (tie)" : "")
             << "\n";
    }
    cout << cmpSep << "\n";
    cout << "  Wins — White: " << winCount[0]
         << "  Spin: "         << winCount[1]
         << "  Coupling: "     << winCount[2] << "\n";

    // ── Per-noise-mode CSV ────────────────────────────────────────────────────
    for (int nm = 0; nm < 3; ++nm) {
        string csvPath = string("sbm_") + NOISE_SHORT[nm] + "_results.csv";
        ofstream csv(csvPath);
        if (!csv.is_open()) { cerr << "[WARN] Could not write " << csvPath << "\n"; continue; }
        csv << "Graph,Nodes,Edges,BestCut,PctEdges,Mean,Std,TotalMs,AvgMsPerTrial\n";
        for (auto& row : allResults) {
            const Result& r = row[nm];
            double pct = (r.m > 0) ? 100.0 * r.bestCut / r.m : 0.0;
            csv << fixed << r.name << "," << r.n << "," << r.m << ","
                << r.bestCut << ","
                << setprecision(4) << pct     << ","
                << setprecision(2) << r.meanCut << ","
                << setprecision(2) << r.stdCut  << ","
                << setprecision(0) << r.totalMs << ","
                << setprecision(0) << r.totalMs / r.trials << "\n";
        }
        csv.close();
        cout << "  Saved: " << csvPath << "\n";
    }

    // ── Comparison CSV ────────────────────────────────────────────────────────
    {
        string csvPath = "sbm_noise_comparison.csv";
        ofstream csv(csvPath);
        if (csv.is_open()) {
            csv << "Graph,Nodes,Edges,"
                << "White_BestCut,White_Pct,White_Mean,White_Std,White_TotalMs,"
                << "Spin_BestCut,Spin_Pct,Spin_Mean,Spin_Std,Spin_TotalMs,"
                << "Coupling_BestCut,Coupling_Pct,Coupling_Mean,Coupling_Std,Coupling_TotalMs,"
                << "BestMode\n";
            for (auto& row : allResults) {
                int cuts[3] = { row[0].bestCut, row[1].bestCut, row[2].bestCut };
                int bestNM  = (int)(max_element(cuts, cuts + 3) - cuts);
                csv << fixed << row[0].name << "," << row[0].n << "," << row[0].m;
                for (int nm = 0; nm < 3; ++nm) {
                    const Result& r = row[nm];
                    double pct = (r.m > 0) ? 100.0 * r.bestCut / r.m : 0.0;
                    csv << "," << r.bestCut
                        << "," << setprecision(4) << pct
                        << "," << setprecision(2) << r.meanCut
                        << "," << setprecision(2) << r.stdCut
                        << "," << setprecision(0) << r.totalMs;
                }
                csv << "," << NOISE_SHORT[bestNM] << "\n";
            }
            csv.close();
            cout << "  Saved: " << csvPath << "\n";
        } else {
            cerr << "[WARN] Could not write " << csvPath << "\n";
        }
    }

    cout << "\n======================================================\n";
    cout << "  Done. Solved " << solved << " graph(s)";
    if (failed) cout << ", failed to load " << failed;
    cout << ".\n";
    return (failed > 0) ? 1 : 0;
}
