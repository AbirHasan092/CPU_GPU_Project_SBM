/**
 * Random Graph Generator
 *
 * Model:
 *   p ~ Uniform(0, 1)  — edge probability drawn from a uniform distribution
 *   For each pair (i,j) with i < j:  edge included iff Uniform(0,1) < p
 *
 * Parameters (edit below):
 *   NUM_NODES   — number of vertices n
 *   NUM_GRAPHS  — how many independent graphs to generate
 *
 * Output: one file per graph, named G1.txt, G2.txt, ..., GK.txt
 * Format per file:
 *   n <num_nodes>
 *   p <edge_probability>
 *   m <num_edges>
 *   <u> <v>       (one edge per line, 0-indexed vertices)
 *   ...
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <iomanip>
#include <string>

// ── User-configurable parameters ─────────────────────────────────────────────

const int NUM_NODES  = 5000;   // number of nodes n
const int NUM_GRAPHS = 10;    // number of graphs to generate

// ─────────────────────────────────────────────────────────────────────────────

struct Edge {
    int u, v;
    Edge(int uu, int vv) : u(uu), v(vv) {}
};

struct Graph {
    int n;
    double p;
    std::vector<Edge> edges;

    int num_edges() const { return static_cast<int>(edges.size()); }

    void print_summary(int id) const {
        std::cout << "Graph G" << id
                  << "  |  n=" << n
                  << "  p=" << std::fixed << std::setprecision(4) << p
                  << "  edges=" << num_edges()
                  << "  (expected=" << std::setprecision(1)
                  << p * n * (n - 1) / 2.0 << ")\n";
    }
};

// Generate one G(n, p) graph where p ~ Uniform(0,1)
Graph generate(int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // Sample p from Uniform(0,1)
    //double p = uniform(rng);
    double p = 0.001; // fixed p for testing

    Graph G;
    G.n = n;
    G.p = p;

    // Include edge (i,j) iff Uniform(0,1) < p
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (uniform(rng) < p)
                G.edges.push_back(Edge(i, j));

    return G;
}

// Export a single graph to its own file (G1.txt, G2.txt, ...)
// id is 1-based to match the filename
void export_graph(const Graph& G, int id) {
    std::string filename = "G" + std::to_string(id) + "_5000n_0.1percent.txt";
    std::ofstream f(filename);
    if (!f) { std::cerr << "Cannot open " << filename << "\n"; return; }

    f << "n " << G.n << "\n";
    f << "p " << std::fixed << std::setprecision(6) << G.p << "\n";
    f << "m " << G.num_edges() << "\n";
    for (auto& e : G.edges)
        f << e.u << " " << e.v << "\n";

    std::cout << "  -> saved to '" << filename << "'\n";
}

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());  // seed from hardware entropy

    for (int i = 0; i < NUM_GRAPHS; ++i) {
        Graph G = generate(NUM_NODES, rng);
        G.print_summary(i + 1);
        export_graph(G, i + 1);
    }

    std::cout << "\nDone. Generated " << NUM_GRAPHS << " file(s).\n";
    return 0;
}
