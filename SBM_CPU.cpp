/**
 * SBM_GPU — CUDA-Accelerated Simulated Bifurcation Machine (MaxCut)
 *
 * Based on SBM2.cpp.  The per-oscillator inner loop is mapped 1-thread-per-node
 * onto the GPU.  The graph is stored in CSR format on the device.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_75 -o SBM_GPU SBM_CPU.cpp -lcurand
 *   (If nvcc rejects .cpp, rename the file to SBM_CPU.cu first.)
 *
 * Usage:  same flags as SBM2.cpp
 *   ./SBM_GPU [--flag value ...] [file1.txt file2.txt ...]
 *
 * Requires: NVIDIA GPU + CUDA Toolkit >= 11.0
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
#include <cmath>
#include <random>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <sstream>
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

bool loadGraph(const string& filename, Graph& g) {
    ifstream f(filename);
    if (!f.is_open()) return false;

    bool dimacs = false, has_n = false;
    vector<pair<int,int>> raw;
    string line;

    while (getline(f, line)) {
        stripCR(line);
        if (line.empty() || line[0] == '#') continue;
        if (line[0] == 'c' && (line.size()==1 || isspace((unsigned char)line[1]))) continue;

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
        if (!has_n && isInt(tok)) { g.n = stoi(tok); ss >> g.m; has_n = true; continue; }
        if (isInt(tok))           { int u = stoi(tok), v; if (ss >> v) raw.push_back({u, v}); }
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
    int* d_rowPtr = nullptr;   // size n+1
    int* d_colIdx = nullptr;   // size 2*m  (undirected edges stored both ways)

    void upload(const Graph& g) {
        n = g.n; m = g.m;
        int nnz = 2 * m;

        // Build CSR on host
        vector<int> rowPtr(n + 1, 0);
        vector<int> colIdx;
        colIdx.reserve(nnz);
        for (int i = 0; i < n; ++i) rowPtr[i+1] = rowPtr[i] + (int)g.adj[i].size();
        for (int i = 0; i < n; ++i) for (int j : g.adj[i]) colIdx.push_back(j);

        // Upload to device
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
    double dt       = 0.10;   // Euler step size
    int    steps    = -1;     // time steps per restart  (-1 = auto)
    double a0       = 1.0;    // pump amplitude
    double Delta    = 1.0;    // frequency parameter
    double c        = 0.5;    // Kerr nonlinear coefficient
    double xi       = -1.0;   // coupling strength        (<0 = auto)
    double noise    = 0.02;   // Gaussian noise sigma
    int    restarts = -1;     // restarts per trial       (-1 = auto)
    int    trials   = 5;
    bool   postLS   = true;
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

// ─── CUDA kernels ─────────────────────────────────────────────────────────────
//  Thread layout: one thread per oscillator (node i).
//  Block size 256 — works well across all modern GPU architectures.

static const int BLOCK = 256;

// Kernel 1 — Initialise one curandState per oscillator
__global__ void k_initRNG(curandState* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, (unsigned long long)i, 0, &states[i]);
}

// Kernel 2 — Randomise oscillator positions x[i], momenta y[i] in [-0.02, 0.02]
__global__ void k_initOsc(double* x, double* y, curandState* states, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = 0.04 * (double)curand_uniform(&states[i]) - 0.02;
    y[i] = 0.04 * (double)curand_uniform(&states[i]) - 0.02;
}

// Kernel 3 — bSB coupling field: h[i] = sum_{j in N(i)} x[j]
__global__ void k_couplingBSB(const double* __restrict__ x,
                               const int*    __restrict__ rowPtr,
                               const int*    __restrict__ colIdx,
                               double*       h, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s = 0.0;
    int start = rowPtr[i], end = rowPtr[i+1];
    for (int k = start; k < end; ++k) s += x[colIdx[k]];
    h[i] = s;
}

// Kernel 4 — dSB coupling field: h[i] = sum_{j in N(i)} sign(x[j])
__global__ void k_couplingDSB(const double* __restrict__ x,
                               const int*    __restrict__ rowPtr,
                               const int*    __restrict__ colIdx,
                               double*       h, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s = 0.0;
    int start = rowPtr[i], end = rowPtr[i+1];
    for (int k = start; k < end; ++k) s += (x[colIdx[k]] >= 0.0) ? 1.0 : -1.0;
    h[i] = s;
}

// Kernel 5 — Momentum update (Euler step for y)
//   y[i] += dt * ( -(a0 - pump)*x[i]  -  a0*c*x[i]^3  +  xi*h[i] )
//         + nf * N(0,1)                          [Langevin noise, nf = sigma*sqrt(dt)]
__global__ void k_stepY(double* y, const double* x, const double* h,
                         double pump, double a0, double c, double xi,
                         double dt,   double nf,
                         curandState* states, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi3 = x[i] * x[i] * x[i];
    y[i] += dt * (-(a0 - pump) * x[i]  -  a0 * c * xi3  +  xi * h[i]);
    if (nf > 0.0)
        y[i] += nf * (double)curand_normal(&states[i]);
}

// Kernel 6 — Position update + wall clamp
//   x[i] += dt * delta * y[i]
//   if |x[i]| > 1: clamp x to ±1, zero momentum
__global__ void k_stepX(double* x, double* y, double dt, double delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += dt * delta * y[i];
    if (x[i] >  1.0) { x[i] =  1.0; y[i] = 0.0; }
    if (x[i] < -1.0) { x[i] = -1.0; y[i] = 0.0; }
}

// Kernel 7 — Binarize continuous x to ±1 spin assignment
__global__ void k_binarize(const double* x, int* spins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) spins[i] = (x[i] >= 0.0) ? 1 : -1;
}

// ─── GPU SBM solver — one variant (bSB=0, dSB=1) ────────────────────────────
vector<int> solveGPU(const Graph& g, const GPUGraph& gg,
                      const SBMParams& p, int mode,
                      unsigned long long seed)
{
    int n      = g.n;
    int blocks = (n + BLOCK - 1) / BLOCK;

    // Auto-scale coupling strength: xi = 0.7 / max_degree
    double xi = p.xi;
    if (xi < 0.0) {
        int mx = 0;
        for (int i = 0; i < n; ++i) mx = max(mx, (int)g.adj[i].size());
        xi = (mx > 0) ? 0.7 / mx : 0.7;
    }
    double nf = p.noise * sqrt(p.dt);   // Langevin scale: sigma * sqrt(dt)

    // ── Allocate device arrays ──────────────────────────────────────────────
    double*      d_x;       // oscillator positions
    double*      d_y;       // oscillator momenta
    double*      d_h;       // coupling field
    int*         d_spins;   // binarized output
    curandState* d_states;  // per-thread RNG states

    CUDA_CHECK(cudaMalloc(&d_x,      n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y,      n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_h,      n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_spins,  n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));

    // ── Initialise RNG once per trial (states persist across restarts) ──────
    k_initRNG<<<blocks, BLOCK>>>(d_states, seed, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<int> bestSpins(n, 1);
    int         bestCut = -1;
    vector<int> h_spins(n);

    // ── Restart loop ─────────────────────────────────────────────────────────
    for (int run = 0; run < p.restarts; ++run) {

        // Fresh random oscillator state for this restart
        k_initOsc<<<blocks, BLOCK>>>(d_x, d_y, d_states, n);

        // ── Time-stepping loop ───────────────────────────────────────────────
        for (int t = 0; t < p.steps; ++t) {
            double pump = p.a0 * (double)t / p.steps;   // linear ramp 0 → a0

            // Coupling field (bSB uses x[j], dSB uses sign(x[j]))
            if (mode == 0)
                k_couplingBSB<<<blocks, BLOCK>>>(d_x, gg.d_rowPtr, gg.d_colIdx, d_h, n);
            else
                k_couplingDSB<<<blocks, BLOCK>>>(d_x, gg.d_rowPtr, gg.d_colIdx, d_h, n);

            // Momentum Euler step  (kernels in default stream → sequenced)
            k_stepY<<<blocks, BLOCK>>>(d_y, d_x, d_h,
                                        pump, p.a0, p.c, xi,
                                        p.dt, nf, d_states, n);
            // Position Euler step + wall clip
            k_stepX<<<blocks, BLOCK>>>(d_x, d_y, p.dt, p.Delta, n);
        }

        // ── Extract spins → CPU → optional local search ─────────────────────
        k_binarize<<<blocks, BLOCK>>>(d_x, d_spins, n);
        CUDA_CHECK(cudaMemcpy(h_spins.data(), d_spins,
                              n * sizeof(int), cudaMemcpyDeviceToHost));

        if (p.postLS) localSearch(g, h_spins);   // greedy 1-flip (CPU)
        int cut = evalCut(g, h_spins);
        if (cut > bestCut) { bestCut = cut; bestSpins = h_spins; }
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_h);
    cudaFree(d_spins); cudaFree(d_states);
    return bestSpins;
}

// ─── Per-graph driver ─────────────────────────────────────────────────────────
void solveGraph(const Graph& g, const GPUGraph& gg, const SBMParams& base) {
    SBMParams p   = scaleParams(base, g.n);
    string    info = "nodes=" + to_string(g.n) + "  edges=" + to_string(g.m);
    if (g.p > 0) info += "  p=" + to_string(g.p).substr(0, 4);
    string cfg = "steps=" + to_string(p.steps) + "  restarts=" + to_string(p.restarts)
               + "  trials=" + to_string(p.trials);

    cout << "\n+----------------------------------------------------+\n";
    cout << "|  " << left << setw(50) << g.name << "|\n";
    cout << "|  " << left << setw(50) << info   << "|\n";
    cout << "|  " << left << setw(50) << cfg    << "|\n";
    cout << "+----------------------------------------------------+\n";
    cout << fixed << setprecision(2);
    cout << "  Trial  |  bSB cut  |  dSB cut  |  Best  |  Time (ms)\n";
    cout << "  -------+-----------+-----------+--------+-----------\n";

    mt19937_64   seedGen(random_device{}());
    vector<int>  bestSpins;
    int          overallBest = -1;
    double       totalTime   = 0.0;
    vector<int>  trialBests(p.trials);

    for (int tr = 0; tr < p.trials; ++tr) {
        unsigned long long seed = seedGen();
        auto t0 = chrono::high_resolution_clock::now();

        auto sb = solveGPU(g, gg, p, 0, seed);                   // bSB
        auto sd = solveGPU(g, gg, p, 1, seed ^ 0xDEADBEEFULL);   // dSB (different seed)

        double elapsed = chrono::duration<double>(
                             chrono::high_resolution_clock::now() - t0).count();
        totalTime += elapsed;

        int cb = evalCut(g, sb), cd = evalCut(g, sd), best = max(cb, cd);
        trialBests[tr] = best;

        cout << "    " << setw(3) << (tr+1) << "    |"
             << "   " << setw(6) << cb      << "    |"
             << "   " << setw(6) << cd      << "    |"
             << "  "  << setw(6) << best    << "  |"
             << "  "  << elapsed * 1000     << "\n";

        if (best > overallBest) { overallBest = best; bestSpins = (cb >= cd) ? sb : sd; }
    }

    int    mn   = *min_element(trialBests.begin(), trialBests.end());
    int    mx   = *max_element(trialBests.begin(), trialBests.end());
    double mean = accumulate(trialBests.begin(), trialBests.end(), 0.0) / p.trials;
    double sq   = 0.0; for (int c : trialBests) sq += (c - mean) * (c - mean);
    double stdv = sqrt(sq / p.trials);
    double ratio = (g.m > 0) ? 100.0 * mx / g.m : 0.0;

    cout << "  -------+-----------+-----------+--------+-----------\n";
    cout << "  min="  << mn << "  max=" << mx << "  mean=" << mean
         << "  std="  << setprecision(1) << stdv << "\n";
    cout << setprecision(2);
    cout << "  Best cut = " << mx << "  (" << ratio << "% of all edges)\n";
    cout << "  Total time = " << totalTime * 1000 << " ms"
         << "  (avg " << totalTime * 1000 / p.trials << " ms/trial)\n";

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
}

// ─── CLI argument parser ──────────────────────────────────────────────────────
static void printHelp(const char* prog) {
    cout <<
"Usage:  " << prog << " [--flag value ...] [file1.txt file2.txt ...]\n"
"\nEuler integration:\n"
"  --dt       <float>   Step size (convergence rate)     default: 0.10\n"
"  --steps    <int>     Time steps per restart           default: auto\n"
"\nSBM physics:\n"
"  --a0       <float>   Pump amplitude                   default: 1.0\n"
"  --c        <float>   Kerr nonlinear coefficient       default: 0.5\n"
"  --xi       <float>   Coupling strength (-1=auto)      default: auto\n"
"  --delta    <float>   Frequency parameter              default: 1.0\n"
"\nNoise (stochastic SB):\n"
"  --noise    <float>   Gaussian noise std per step      default: 0.02\n"
"                       Injected as sigma*sqrt(dt)*N(0,1) into momentum.\n"
"\nSearch:\n"
"  --restarts <int>     Restarts per trial               default: auto\n"
"  --trials   <int>     Independent trials per graph     default: 5\n"
"  --no-ls              Disable greedy 1-flip local search\n"
"\nFiles:\n"
"  Positional args are graph file paths.\n"
"  Default (no files given): all G*_5000n_0.1percent.txt in current directory.\n"
"\nExamples:\n"
"  " << prog << " G1_5000n_0.1percent.txt\n"
"  " << prog << " --steps 3000 --noise 0.01 G1_5000n_0.1percent.txt\n"
"  " << prog << " --trials 10 --no-ls G1_5000n_0.1percent.txt G2_5000n_0.1percent.txt\n";
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

        if      (a == "--dt")       p.dt       = stod(need());
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
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    cout << "======================================================\n";
    cout << "  SBM_GPU -- CUDA Simulated Bifurcation Machine (MaxCut)\n";
    cout << "  GPU: " << prop.name
         << "  (" << prop.multiProcessorCount << " SMs, "
         << prop.totalGlobalMem / 1024 / 1024 << " MB VRAM)\n";
    cout << "  Variants: bSB (ballistic)  |  dSB (discrete)\n";
    cout << "  Reference: Goto et al., Science Advances 7(6), 2021\n";
    cout << "======================================================\n";

    vector<string> files;
    SBMParams params = parseArgs(argc, argv, files);

    // Default: glob G*_5000n_0.1percent.txt
    if (files.empty()) {
        FILE* ls = popen("ls G*_5000n_0.1percent.txt 2>/dev/null", "r");
        if (ls) {
            char buf[512];
            while (fgets(buf, sizeof(buf), ls)) {
                string s(buf); stripCR(s);
                if (!s.empty() && s.back() == '\n') s.pop_back();
                stripCR(s);
                if (!s.empty()) files.push_back(s);
            }
            pclose(ls);
        }
        if (files.empty()) {
            cerr << "[ERROR] No G*_5000n_0.1percent.txt files found. "
                    "Pass files explicitly or use --help.\n";
            return 1;
        }
        cout << "\nDefaulting to " << files.size()
             << " file(s) matching G*_5000n_0.1percent.txt\n";
    }

    // Print active parameter set
    SBMParams ref = scaleParams(params, 300);
    cout << "\nActive parameters:\n";
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

    int solved = 0, failed = 0;
    for (auto& fname : files) {
        Graph g;
        size_t sl  = fname.find_last_of("/\\");
        string base = (sl == string::npos) ? fname : fname.substr(sl + 1);
        size_t dot  = base.rfind('.');
        g.name      = (dot == string::npos) ? base : base.substr(0, dot);

        if (!loadGraph(fname, g)) {
            cerr << "\n[ERROR] Cannot open '" << fname << "'.\n";
            ++failed; continue;
        }

        // Build CSR and upload graph to GPU
        GPUGraph gg;
        gg.upload(g);

        solveGraph(g, gg, params);

        gg.free_();
        ++solved;
    }

    cout << "\n======================================================\n";
    cout << "  Done. Solved " << solved << " graph(s)";
    if (failed) cout << ", failed to load " << failed;
    cout << ".\n";
    return (failed > 0) ? 1 : 0;
}