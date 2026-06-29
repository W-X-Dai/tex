#include <bits/stdc++.h>
using namespace std;
using ll = long long;

const ll MOD = 1000000007;

tuple<ll, ll, ll> egcd(ll a, ll b) {
    if (a == 0) return {b, 0, 1};
    auto [g, x1, y1] = egcd(b % a, a);
    ll x = y1 - (b / a) * x1;
    ll y = x1;
    return {g, x, y};
}

ll mod_inverse(ll a, ll m) {
    auto [g, x, y] = egcd(a, m);
    if (g != 1) return -1;
    return (x % m + m) % m;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int T;
    cin >> T;
    for (int t = 0; t < T; t++) {
        ll a, b, c;
        cin >> a >> b >> c;
        
        a %= MOD;
        b %= MOD;
        c %= MOD;
        
        ll inv = mod_inverse(c, MOD);
        if (inv == -1) {
            cout << -1 << '\n';
        } else {
            ll ans = a * b % MOD * inv % MOD;
            cout << ans << '\n';
        }
    }
    return 0;
}