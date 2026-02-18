#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <cctype>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;

/* ---------- single-pair LCS (longest common substring) ---------- */
std::pair<int, std::string>
lcs_pair(const std::string &S_in, const std::string &T_in) {
    const size_t m = S_in.size(), n = T_in.size();
    std::vector<int> prev(n + 1, 0), curr(n + 1, 0);

    int best_len = 0, best_pos = -1;           // end-index in S
    auto lower = [](char c){ return std::tolower(static_cast<unsigned char>(c)); };

    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            curr[j] = (lower(S_in[i-1]) == lower(T_in[j-1]))
                       ? prev[j-1] + 1 : 0;
            if (curr[j] > best_len) {
                best_len = curr[j];
                best_pos = static_cast<int>(i);
            }
        }
        std::swap(prev, curr);
        std::fill(curr.begin(), curr.end(), 0);
    }

    std::string best_pattern = (best_len>0)
         ? S_in.substr(best_pos - best_len, best_len)
         : "";

    return {best_len, best_pattern};
}

/* ---------- batched pairwise computation ---------- */
py::tuple
longest_match_distance(const std::vector<std::string> &str1s,
                       const std::vector<std::string> &str2s)
{
    std::vector<std::vector<double>>  longest_string(str1s.size());
    std::vector<std::vector<std::string>> longest_pattern(str1s.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < str1s.size(); ++i) {
        auto &row_dist  = longest_string[i];
        auto &row_pat   = longest_pattern[i];
        row_dist.reserve(str2s.size());
        row_pat.reserve(str2s.size());

        for (const auto &s2 : str2s) {
            auto [len, pat] = lcs_pair(str1s[i], s2);
            row_dist .push_back(1.0 - static_cast<double>(len) /
                                static_cast<double>(str1s[i].size()));
            row_pat .push_back(std::move(pat));
        }
    }
    return py::make_tuple(longest_string, longest_pattern);
}

/* ---------- pybind11 glue ---------- */
PYBIND11_MODULE(_lcs_ext, m) {
    m.doc() = "Fast longest-common-substring utilities";
    m.def("longest_match_distance", &longest_match_distance,
          py::arg("str1s"), py::arg("str2s"),
          R"pbdoc(
    Compute 1-LCS-distance and best pattern for every (s1, s2) pair.

    Returns
    -------
    distances : List[List[float]]
    patterns  : List[List[str]]
)pbdoc");
}
