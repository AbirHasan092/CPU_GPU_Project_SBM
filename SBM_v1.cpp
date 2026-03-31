/**
 * SBM2 — Simulated Bifurcation Machine with User-Controlled Parameters
 *
 * All SBM model parameters, noise level, and Euler step size are
 * configurable via command-line flags.
 *
 * Usage:
 *   ./SBM2 [--flag value ...] [file1.txt file2.txt ...]
 *
 * Flags:
 *   --dt       <float>  Euler step size Δt            (default: 0.10)
 *   --steps    <int>    Time steps per restart        (default: auto)
 *   --restarts <int>    Restarts per trial            (default: auto)
 *   --trials   <int>    Independent trials per graph  (default: 5)
 *   --a0       <float>  Pump amplitude a0             (default: 1.0)
 *   --c        <float>  Kerr nonlinear coefficient    (default: 0.5)
 *   --xi       <float>  Coupling strength xi          (default: auto)
 *   --delta    <float>  Frequency parameter delta     (default: 1.0)
 *   --noise    <float>  Gaussian noise std sigma      (default: 0.0)
 *   --no-ls             Disable greedy local search
 *   --help              Print this message and exit
 *
 * Noise injection (stochastic SB):
 *   Each Euler step: y_i += sigma * sqrt(dt) * N(0,1)
 *   sigma = 0  -> deterministic SBM (original behaviour)
 *   sigma > 0  -> stochastic SBM; helps escape shallow local optima
 *
 * Euler convergence:
 *   Smaller --dt gives finer integration but requires more --steps to
 *   cover the same pump range. Rule of thumb: steps * dt ~ 300 for n~300.
 *
 * Reference:
 *   Goto et al., "High-performance combinatorial optimization based on
 *   classical mechanics", Science Advances 7(6), 2021.
 */

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
#include <limits>

using namespace std;

// ─── Utilities ───────────────────────────────────────────────────────────────

static void stripCR(string& s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
}
static bool isInt(const string& s) {
    if (s.empty()) return false;
    size_t i = (s[0]=='-') ? 1 : 0;
    for (; i < s.size(); ++i) if (!isdigit((unsigned char)s[i])) return false;
    return true;
}

// ─── Graph ───────────────────────────────────────────────────────────────────

struct Graph {
    string name;
    int    n=0, m=0;
    double p=0.0;
    vector<vector<int> > adj;
};

bool loadGraph(const string& filename, Graph& g) {
    ifstream f(filename);
    if (!f.is_open()) return false;

    bool dimacs=false, has_n=false;
    vector<pair<int,int> > raw;

    string line;
    while (getline(f, line)) {
        stripCR(line);
        if (line.empty() || line[0]=='#') continue;
        if (line[0]=='c' && (line.size()==1 || isspace((unsigned char)line[1]))) continue;

        istringstream ss(line); string tok; ss >> tok;

        if (tok=="n")                    { ss>>g.n; has_n=true; continue; }
        if (tok=="p" && !has_n)          { string t2; ss>>t2;
                                           if(t2=="edge"){ss>>g.n>>g.m; has_n=dimacs=true;}
                                           continue; }
        if (tok=="p" && has_n)           { continue; }
        if (tok=="m")                    { ss>>g.m; continue; }
        if (dimacs && (tok=="e"||tok=="E")) { int u,v; ss>>u>>v; raw.push_back(pair<int,int>(u-1,v-1)); continue; }
        if (!has_n && isInt(tok))        { g.n=stoi(tok); ss>>g.m; has_n=true; continue; }
        if (isInt(tok))                  { int u=stoi(tok),v; if(ss>>v) raw.push_back(pair<int,int>(u,v)); }
    }

    if (!has_n) {
        int mx=-1; for(auto&e:raw) mx=max(mx,max(e.first,e.second));
        if (mx<0) return false; g.n=mx+1;
    }
    if (g.n<=0) return false;
    g.adj.assign(g.n, vector<int>());
    for (auto&e:raw) {
        int u=e.first,v=e.second;
        if(u<0||v<0||u>=g.n||v>=g.n||u==v) continue;
        g.adj[u].push_back(v); g.adj[v].push_back(u);
    }
    int s=0; for(int i=0;i<g.n;++i) s+=(int)g.adj[i].size();
    g.m=s/2;
    return g.n>0 && g.m>0;
}

// ─── MaxCut eval + local search ──────────────────────────────────────────────

int evalCut(const Graph& g, const vector<int>& spins) {
    int cut=0;
    for(int u=0;u<g.n;++u) for(int v:g.adj[u]) if(v>u&&spins[u]!=spins[v]) ++cut;
    return cut;
}

void localSearch(const Graph& g, vector<int>& spins) {
    bool imp=true;
    while(imp) { imp=false;
        for(int i=0;i<g.n;++i) {
            int gain=0; for(int j:g.adj[i]) gain+=(spins[i]==spins[j])?1:-1;
            if(gain>0){spins[i]=-spins[i];imp=true;}
        }
    }
}

// ─── SBM parameters ──────────────────────────────────────────────────────────

struct SBMParams {
    // Euler integration
    double dt       = 0.10;   // step size  -- controls convergence rate
    int    steps    = -1;     // time steps (-1 = auto-scale)

    // Physics
    double a0       = 1.0;    // pump amplitude
    double Delta    = 1.0;    // frequency parameter
    double c        = 0.5;    // Kerr nonlinear coefficient
    double xi       = -1.0;   // coupling strength (<0 = auto)

    // Noise
    double noise    = 0.02;    // Gaussian noise std sigma

    // Search
    int    restarts = -1;     // restarts per trial (-1 = auto)
    int    trials   = 5;
    bool   postLS   = true;
};

SBMParams scaleParams(const SBMParams& base, int n) {
    SBMParams p = base;
    if (p.steps   < 0) p.steps   = max(1000,(int)(1000.0*pow((double)n/50.0,0.6)));
    if (p.restarts < 0) p.restarts = max(5,(int)(20.0*pow(50.0/max(n,50),0.5)));
    return p;
}

// ─── bSB solver ──────────────────────────────────────────────────────────────

vector<int> solveBSB(const Graph& g, const SBMParams& p, mt19937& rng) {
    int n=g.n;
    double xi=p.xi;
    if(xi<0){int mx=0; for(int i=0;i<n;++i) mx=max(mx,(int)g.adj[i].size()); xi=(mx>0)?0.7/mx:0.7;}

    uniform_real_distribution<double> init(-0.02,0.02);
    normal_distribution<double>       gauss(0.0,1.0);
    double nf = p.noise * sqrt(p.dt);   // sigma * sqrt(dt) per step

    vector<int>    best(n,1); int bestCut=-1;
    vector<double> x(n),y(n),h(n);

    for(int run=0; run<p.restarts; ++run) {
        for(int i=0;i<n;++i){x[i]=init(rng);y[i]=init(rng);}

        for(int t=0; t<p.steps; ++t) {
            double pump = p.a0*(double)t/p.steps;

            fill(h.begin(),h.end(),0.0);
            for(int i=0;i<n;++i) for(int j:g.adj[i]) h[i]+=x[j];

            for(int i=0;i<n;++i) {
                y[i] += p.dt*(-(p.a0-pump)*x[i] - p.a0*p.c*x[i]*x[i]*x[i] + xi*h[i]);
                if(nf>0.0) y[i] += nf*gauss(rng);   // <-- noise injection
            }
            for(int i=0;i<n;++i) {
                x[i] += p.dt*p.Delta*y[i];
                if(x[i]> 1.0){x[i]= 1.0;y[i]=0.0;}
                if(x[i]<-1.0){x[i]=-1.0;y[i]=0.0;}
            }
        }

        vector<int> spins(n);
        for(int i=0;i<n;++i) spins[i]=(x[i]>=0)?+1:-1;
        if(p.postLS) localSearch(g,spins);
        int cut=evalCut(g,spins);
        if(cut>bestCut){bestCut=cut;best=spins;}
    }
    return best;
}

// ─── dSB solver ──────────────────────────────────────────────────────────────

vector<int> solveDSB(const Graph& g, const SBMParams& p, mt19937& rng) {
    int n=g.n;
    double xi=p.xi;
    if(xi<0){int mx=0; for(int i=0;i<n;++i) mx=max(mx,(int)g.adj[i].size()); xi=(mx>0)?0.7/mx:0.7;}

    uniform_real_distribution<double> init(-0.02,0.02);
    normal_distribution<double>       gauss(0.0,1.0);
    double nf = p.noise * sqrt(p.dt);

    vector<int>    best(n,1); int bestCut=-1;
    vector<double> x(n),y(n),h(n);

    for(int run=0; run<p.restarts; ++run) {
        for(int i=0;i<n;++i){x[i]=init(rng);y[i]=init(rng);}

        for(int t=0; t<p.steps; ++t) {
            double pump = p.a0*(double)t/p.steps;

            fill(h.begin(),h.end(),0.0);
            for(int i=0;i<n;++i) for(int j:g.adj[i]) h[i]+=(x[j]>=0)?1.0:-1.0;

            for(int i=0;i<n;++i) {
                y[i] += p.dt*(-(p.a0-pump)*x[i] - p.a0*p.c*x[i]*x[i]*x[i] + xi*h[i]);
                if(nf>0.0) y[i] += nf*gauss(rng);   // <-- noise injection
            }
            for(int i=0;i<n;++i) {
                x[i] += p.dt*p.Delta*y[i];
                if(x[i]> 1.0){x[i]= 1.0;y[i]=0.0;}
                if(x[i]<-1.0){x[i]=-1.0;y[i]=0.0;}
            }
        }

        vector<int> spins(n);
        for(int i=0;i<n;++i) spins[i]=(x[i]>=0)?+1:-1;
        if(p.postLS) localSearch(g,spins);
        int cut=evalCut(g,spins);
        if(cut>bestCut){bestCut=cut;best=spins;}
    }
    return best;
}

// ─── Per-graph driver ─────────────────────────────────────────────────────────

void solveGraph(const Graph& g, const SBMParams& base) {
    SBMParams p = scaleParams(base, g.n);
    string info = "nodes="+to_string(g.n)+"  edges="+to_string(g.m);
    if(g.p>0) info+="  p="+to_string(g.p).substr(0,4);
    string cfg  = "steps="+to_string(p.steps)+"  restarts="+to_string(p.restarts)
                + "  trials="+to_string(p.trials);

    cout<<"\n+----------------------------------------------------+\n";
    cout<<"|  "<<left<<setw(50)<<g.name <<"|\n";
    cout<<"|  "<<left<<setw(50)<<info   <<"|\n";
    cout<<"|  "<<left<<setw(50)<<cfg    <<"|\n";
    cout<<"+----------------------------------------------------+\n";
    cout<<fixed<<setprecision(2);
    cout<<"  Trial  |  bSB cut  |  dSB cut  |  Best  |  Time (ms)\n";
    cout<<"  -------+-----------+-----------+--------+-----------\n";

    random_device rd;
    vector<int>   bestSpins; int overallBest=-1; double totalTime=0.0;
    vector<int>   trialBests(p.trials);

    for(int tr=0; tr<p.trials; ++tr) {
        mt19937 rng(rd());
        auto t0=chrono::high_resolution_clock::now();

        auto sb=solveBSB(g,p,rng);
        auto sd=solveDSB(g,p,rng);

        double elapsed=chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count();
        totalTime+=elapsed;

        int cb=evalCut(g,sb), cd=evalCut(g,sd), best=max(cb,cd);
        trialBests[tr]=best;

        cout<<"    "<<setw(3)<<(tr+1)<<"    |"
            <<"   "<<setw(6)<<cb<<"    |"
            <<"   "<<setw(6)<<cd<<"    |"
            <<"  " <<setw(6)<<best<<"  |"
            <<"  " <<elapsed*1000<<"\n";

        if(best>overallBest){overallBest=best; bestSpins=(cb>=cd)?sb:sd;}
    }

    int    mn=*min_element(trialBests.begin(),trialBests.end());
    int    mx=*max_element(trialBests.begin(),trialBests.end());
    double mean=accumulate(trialBests.begin(),trialBests.end(),0.0)/p.trials;
    double sq=0; for(int c:trialBests) sq+=(c-mean)*(c-mean);
    double stdv=sqrt(sq/p.trials);
    double ratio=(g.m>0)?100.0*mx/g.m:0.0;

    cout<<"  -------+-----------+-----------+--------+-----------\n";
    cout<<"  min="<<mn<<"  max="<<mx<<"  mean="<<mean<<"  std="<<setprecision(1)<<stdv<<"\n";
    cout<<setprecision(2);
    cout<<"  Best cut = "<<mx<<"  ("<<ratio<<"% of all edges)\n";
    cout<<"  Total time = "<<totalTime*1000<<" ms"
        <<"  (avg "<<totalTime*1000/p.trials<<" ms/trial)\n";

    // Partition
    if(g.n<=100) {
        cout<<"  Best partition:\n    S0(+1): {";
        bool first=true;
        for(int i=0;i<g.n;++i) if(bestSpins[i]==+1){if(!first)cout<<",";cout<<i;first=false;}
        cout<<"}\n    S1(-1): {"; first=true;
        for(int i=0;i<g.n;++i) if(bestSpins[i]==-1){if(!first)cout<<",";cout<<i;first=false;}
        cout<<"}\n";
    } else {
        int s0=(int)count(bestSpins.begin(),bestSpins.end(),+1);
        cout<<"  Best partition:  S0(+1): "<<s0<<" nodes  |  S1(-1): "<<(g.n-s0)<<" nodes\n";
    }
}

// ─── CLI argument parser ─────────────────────────────────────────────────────

static void printHelp(const char* prog) {
    cout<<
"Usage:  "<<prog<<" [--flag value ...] [file1.txt file2.txt ...]\n"
"\n"
"Euler integration:\n"
"  --dt       <float>   Step size (convergence rate)     default: 0.10\n"
"  --steps    <int>     Time steps per restart           default: auto\n"
"\n"
"SBM physics:\n"
"  --a0       <float>   Pump amplitude                   default: 1.0\n"
"  --c        <float>   Kerr nonlinear coefficient       default: 0.5\n"
"  --xi       <float>   Coupling strength (-1=auto)      default: auto\n"
"  --delta    <float>   Frequency parameter              default: 1.0\n"
"\n"
"Noise (stochastic SB):\n"
"  --noise    <float>   Gaussian noise std per step      default: 0.0\n"
"                       Injected as sigma*sqrt(dt)*N(0,1) into momentum.\n"
"                       Increase to escape shallow local optima.\n"
"\n"
"Search:\n"
"  --restarts <int>     Restarts per trial               default: auto\n"
"  --trials   <int>     Independent trials per graph     default: 5\n"
"  --no-ls              Disable greedy 1-flip local search\n"
"\n"
"Files:\n"
"  Positional args are graph file paths.\n"
"  Default (no files given): all G*_300n_0.2p.txt in current directory.\n"
"\n"
"Examples:\n"
"  "<<prog<<" G1_300n_0.2p.txt\n"
"  "<<prog<<" --dt 0.05 --steps 6000 --noise 0.01 G1_300n_0.2p.txt\n"
"  "<<prog<<" --a0 1.5 --c 0.3 --trials 10 G1_300n_0.2p.txt G2_300n_0.2p.txt\n"
"  "<<prog<<" --noise 0.05 --no-ls G1_300n_0.2p.txt\n";
}

SBMParams parseArgs(int argc, char* argv[], vector<string>& files) {
    SBMParams p;
    for(int i=1; i<argc; ++i) {
        string a=argv[i];
        if(a=="--help"||a=="-h"){printHelp(argv[0]);exit(0);}
        if(a=="--no-ls"){p.postLS=false;continue;}

        if (a=="--dt") {
            if(i+1>=argc){cerr<<"[ERROR] "<<a<<" requires a value.\n";exit(1);}
            p.dt = stod(argv[++i]);
        } else if (a=="--steps") {
            if(i+1>=argc){cerr<<"[ERROR] "<<a<<" requires a value.\n";exit(1);}
            p.steps = stoi(argv[++i]);
        } else if (a=="--restarts") {
            if(i+1>=argc){cerr<<"[ERROR] "<<a<<" requires a value.\n";exit(1);}
            p.restarts = stoi(argv[++i]);
        } else if (a=="--trials") {
            if(i+1>=argc){cerr<<"[ERROR] "<<a<<" requires a value.\n";exit(1);}
            p.trials = stoi(argv[++i]);
        } else if (a=="--a0") {
            if(i+1>=argc){cerr<<"[ERROR] "<<a<<" requires a value.\n";exit(1);}
            p.a0 = stod(argv[++i]);
        } else if (a=="--c") {
            if(i+1>=argc){cerr<<"[ERROR] "<<a<<" requires a value.\n";exit(1);}
            p.c = stod(argv[++i]);
        } else if (a=="--xi") {
            if(i+1>=argc){cerr<<"[ERROR] "<<a<<" requires a value.\n";exit(1);}
            p.xi = stod(argv[++i]);
        } else if (a=="--delta") {
            if(i+1>=argc){cerr<<"[ERROR] "<<a<<" requires a value.\n";exit(1);}
            p.Delta = stod(argv[++i]);
        } else if (a=="--noise") {
            if(i+1>=argc){cerr<<"[ERROR] "<<a<<" requires a value.\n";exit(1);}
            p.noise = stod(argv[++i]);
        } else if (a[0]=='-') {
            cerr<<"[WARN] Unknown flag: "<<a<<"\n";
        } else {
            files.push_back(a);
        }
    }
    return p;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    cout<<"======================================================\n";
    cout<<"  SBM2 -- Simulated Bifurcation Machine (MaxCut)\n";
    cout<<"  Variants: bSB (ballistic)  |  dSB (discrete)\n";
    cout<<"  Reference: Goto et al., Science Advances 7(6), 2021\n";
    cout<<"======================================================\n";

    vector<string> files;
    SBMParams params = parseArgs(argc, argv, files);

    // Default files
    if(files.empty()) {
        FILE* ls=popen("ls G*_5000n_0.1percent.txt 2>/dev/null","r");
        if(ls){char buf[512];
            while(fgets(buf,sizeof(buf),ls)){
                string s(buf); stripCR(s);
                if(!s.empty()&&s.back()=='\n') s.pop_back(); stripCR(s);
                if(!s.empty()) files.push_back(s);
            } pclose(ls);
        }
        if(files.empty()){
            cerr<<"[ERROR] No G*_5000n_0.1percent.txt files found. Pass files explicitly or use --help.\n";
            return 1;
        }
        cout<<"\nDefaulting to "<<files.size()<<" file(s) matching G*_5000n_0.1percent.txt\n";
    }

    // Print active parameter set (show auto-scaled values at n=300 as reference)
    SBMParams ref = scaleParams(params, 300);
    cout<<"\nActive parameters:\n";
    cout<<"  Euler  :  dt="<<params.dt
        <<"  steps=" <<(params.steps<0  ? to_string(ref.steps)+"(auto)"    : to_string(params.steps))
        <<"  restarts="<<(params.restarts<0 ? to_string(ref.restarts)+"(auto)" : to_string(params.restarts))<<"\n";
    cout<<"  Physics:  a0="<<params.a0
        <<"  c="<<params.c
        <<"  xi="<<(params.xi<0?"auto":to_string(params.xi))
        <<"  delta="<<params.Delta<<"\n";
    cout<<"  Noise  :  sigma="<<params.noise
        <<(params.noise==0.0 ? "  (deterministic)" : "  (stochastic SB)")<<"\n";
    cout<<"  Search :  trials="<<params.trials
        <<"  post-LS="<<(params.postLS?"on":"off")<<"\n";

    int solved=0, failed=0;
    for(auto& fname:files) {
        Graph g;
        size_t sl=fname.find_last_of("/\\");
        string base=(sl==string::npos)?fname:fname.substr(sl+1);
        size_t dot=base.rfind('.'); g.name=(dot==string::npos)?base:base.substr(0,dot);
        if(!loadGraph(fname,g)){cerr<<"\n[ERROR] Cannot open '"<<fname<<"'.\n";++failed;continue;}
        solveGraph(g,params); ++solved;
    }

    cout<<"\n======================================================\n";
    cout<<"  Done. Solved "<<solved<<" graph(s)";
    if(failed) cout<<", failed to load "<<failed;
    cout<<".\n";
    return (failed>0)?1:0;
}
