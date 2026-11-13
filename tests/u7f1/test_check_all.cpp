// tests/u7f1/test_check_all.cpp — NativeDenseOverride theorems: checkAll_fails,
// checkAll_length, failed_load_untouched.
// Theory:
//   checkAll_fails : one failing item → whole batch fails
//   checkAll_length: successful batch → length equals input length
//   failed_load_untouched: publish on error leaves table unchanged
#include "u7f1/check.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <optional>

// Simplified checkAll modelled on the Lean definition.
// check: T → Except ε B  (here: Except = optional<string> for error)
template <class T, class B>
std::optional<std::string> checkAll(
        const std::vector<T>& items,
        std::vector<B>& out,
        std::function<std::optional<std::string>(const T&, B&)> check) {
    std::vector<B> acc;
    for (const auto& a : items) {
        B b{};
        auto e = check(a, b);
        if (e) return e;   // first failure aborts
        acc.push_back(b);
    }
    out = std::move(acc);
    return std::nullopt;   // success
}

// Simplified publish: apply only on success.
template <class Tbl, class B>
void publish(Tbl& tbl, const std::optional<std::string>& res,
             const std::vector<B>& xs, std::function<void(Tbl&, const std::vector<B>&)> apply) {
    if (!res) apply(tbl, xs);    // success: apply
    // failure: table unchanged
}

int main() {
    std::puts("=== test_check_all ===");

    // ---- checkAll_fails: one failure → whole batch .error ----
    {
        std::vector<int> items = {1, 2, 3, 4, 5};
        std::vector<int> out;
        // fail on item == 3
        auto err = checkAll<int, int>(items, out, [](const int& x, int& b) -> std::optional<std::string> {
            if (x == 3) return "fail at 3";
            b = x * 2;
            return std::nullopt;
        });
        CHK(err.has_value());         // batch failed
        CHK(out.empty());             // no output on failure
    }

    // ---- checkAll_length: all succeed → length == items.size() ----
    {
        std::vector<int> items = {10, 20, 30};
        std::vector<int> out;
        auto err = checkAll<int, int>(items, out, [](const int& x, int& b) -> std::optional<std::string> {
            b = x; return std::nullopt;
        });
        CHK(!err.has_value());          // success
        CHK(out.size() == items.size()); // checkAll_length
        CHK(out[0] == 10 && out[1] == 20 && out[2] == 30);
    }

    // ---- failed_load_untouched: publish on error leaves table unchanged ----
    {
        int table = 42;
        std::vector<int> xs = {1, 2, 3};
        std::optional<std::string> fail_res = "some error";
        publish<int, int>(table, fail_res, xs,
            [](int& t, const std::vector<int>& v) { t = (int)v.size(); });
        CHK(table == 42);  // table unchanged on failure

        // Successful publish DOES change table
        std::optional<std::string> ok_res = std::nullopt;
        publish<int, int>(table, ok_res, xs,
            [](int& t, const std::vector<int>& v) { t = (int)v.size(); });
        CHK(table == 3);   // table updated on success
    }

    // ---- empty batch succeeds with empty output ----
    {
        std::vector<int> items;
        std::vector<int> out;
        auto err = checkAll<int, int>(items, out, [](const int&, int&) -> std::optional<std::string> {
            return "should not be called";
        });
        CHK(!err.has_value());
        CHK(out.empty());
    }

    return u7f1::finish("TEST_CHECK_ALL");
}

