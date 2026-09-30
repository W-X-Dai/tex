// A. 簡單題 - 標準解
// 支援任意精度整數（含負數）加法
#include <bits/stdc++.h>
using namespace std;

struct BigInt {
    bool neg = false;   // 是否為負數（0 一律視為非負）
    string mag;         // 數值大小的字串（無前導零，最高位在前）
};

// 比較兩個「非負」數字字串的大小：回傳 -1 / 0 / 1
int cmpMag(const string &a, const string &b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

// 兩個非負數字字串相加
string addMag(const string &a, const string &b) {
    string res;
    int i = (int)a.size() - 1, j = (int)b.size() - 1, carry = 0;
    while (i >= 0 || j >= 0 || carry) {
        int x = (i >= 0 ? a[i] - '0' : 0) + (j >= 0 ? b[j] - '0' : 0) + carry;
        carry = x / 10;
        res.push_back('0' + x % 10);
        i--; j--;
    }
    reverse(res.begin(), res.end());
    return res;
}

// 兩個非負數字字串相減，要求 a >= b
string subMag(const string &a, const string &b) {
    string res;
    int i = (int)a.size() - 1, j = (int)b.size() - 1, borrow = 0;
    while (i >= 0) {
        int x = (a[i] - '0') - borrow - (j >= 0 ? b[j] - '0' : 0);
        if (x < 0) { x += 10; borrow = 1; } else borrow = 0;
        res.push_back('0' + x);
        i--; j--;
    }
    while (res.size() > 1 && res.back() == '0') res.pop_back();
    reverse(res.begin(), res.end());
    return res;
}

BigInt parse(const string &s) {
    BigInt b;
    int idx = 0;
    if (!s.empty() && (s[0] == '-' || s[0] == '+')) {
        b.neg = (s[0] == '-');
        idx = 1;
    }
    string mag = s.substr(idx);
    int k = 0;
    while (k + 1 < (int)mag.size() && mag[k] == '0') k++;
    mag = mag.substr(k);
    b.mag = mag;
    if (b.mag == "0") b.neg = false; // 避免 -0
    return b;
}

BigInt add(const BigInt &a, const BigInt &b) {
    BigInt r;
    if (a.neg == b.neg) {
        r.neg = a.neg;
        r.mag = addMag(a.mag, b.mag);
    } else {
        int c = cmpMag(a.mag, b.mag);
        if (c == 0) { r.neg = false; r.mag = "0"; }
        else if (c > 0) { r.neg = a.neg; r.mag = subMag(a.mag, b.mag); }
        else { r.neg = b.neg; r.mag = subMag(b.mag, a.mag); }
    }
    if (r.mag == "0") r.neg = false;
    return r;
}

string toString(const BigInt &b) {
    return (b.neg ? "-" : "") + b.mag;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;
    while (t--) {
        string sa, sb;
        cin >> sa >> sb;
        BigInt a = parse(sa);
        BigInt b = parse(sb);
        BigInt c = add(a, b);
        cout << toString(c) << "\n";
    }
    return 0;
}
