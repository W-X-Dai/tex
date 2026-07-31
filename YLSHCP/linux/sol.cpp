#include <bits/stdc++.h>
using namespace std;

struct Node {
    bool is_dir;
    string name;
    Node* parent;
    map<string, Node*> children;
    Node(bool is_dir, const string& name, Node* parent)
        : is_dir(is_dir), name(name), parent(parent) {}
};

Node* root_node;
Node* cur;

Node* resolvePath(Node* from, const string& path) {
    if (path.empty()) return from;
    Node* node = (path[0] == '/') ? root_node : from;
    stringstream ss(path);
    string seg;
    while (getline(ss, seg, '/')) {
        if (seg.empty() || seg == ".") continue;
        if (seg == "..") {
            if (node->parent) node = node->parent;
        } else {
            node = node->children[seg];
        }
    }
    return node;
}

pair<Node*, string> resolveParentAndName(Node* from, const string& path) {
    size_t pos = path.rfind('/');
    if (pos == string::npos) return {from, path};
    string name = path.substr(pos + 1);
    if (name.empty()) return resolveParentAndName(from, path.substr(0, pos));
    if (name == "." || name == "..") {
        Node* node = resolvePath(from, path);
        return {node->parent, node->name};
    }
    Node* parent = (pos == 0) ? root_node : resolvePath(from, path.substr(0, pos));
    return {parent, name};
}

string getAbsPath(Node* node) {
    if (node == root_node) return "/";
    string path;
    while (node != root_node) { path = "/" + node->name + path; node = node->parent; }
    return path;
}

void deleteTree(Node* node) {
    for (auto& [_, child] : node->children) deleteTree(child);
    delete node;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    root_node = new Node(true, "", nullptr);
    cur = root_node;

    string line;
    while (getline(cin, line)) {
        if (line == "^C") break;
        istringstream iss(line);
        string cmd; iss >> cmd;

        if (cmd == "pwd") {
            cout << getAbsPath(cur) << "\n";
        }
        else if (cmd == "ls") {
            for (auto& [name, _] : cur->children)
                if (name[0] != '.') cout << name << "\n";
        }
        else if (cmd == "la") {
            vector<string> v = {".", ".."};
            for (auto& [name, _] : cur->children) v.push_back(name);
            sort(v.begin(), v.end());
            for (auto& s : v) cout << s << "\n";
        }
        else if (cmd == "mkdir") {
            string name; iss >> name;
            cur->children[name] = new Node(true, name, cur);
        }
        else if (cmd == "touch") {
            string name; iss >> name;
            if (!cur->children.count(name))
                cur->children[name] = new Node(false, name, cur);
        }
        else if (cmd == "cd") {
            string path; iss >> path;
            cur = resolvePath(cur, path);
        }
        else if (cmd == "rm") {
            string arg; iss >> arg;
            bool recursive = (arg == "-r");
            string path; if (recursive) iss >> path; else path = arg;
            Node* t = resolvePath(cur, path);
            // 若 cur 在被刪子樹內，退至父目錄
            for (Node* n = cur; n; n = n->parent)
                if (n == t) { cur = t->parent; break; }
            t->parent->children.erase(t->name);
            deleteTree(t);
        }
        else if (cmd == "mv") {
            string sp, dp; iss >> sp >> dp;
            Node* src = resolvePath(cur, sp);
            Node* dst = resolvePath(cur, dp);
            if (dst && dst->is_dir) {
                src->parent->children.erase(src->name);
                dst->children[src->name] = src;
                src->parent = dst;
            } else {
                auto [par, new_name] = resolveParentAndName(cur, dp);
                src->parent->children.erase(src->name);
                src->name = new_name;
                par->children[new_name] = src;
                src->parent = par;
            }
        }
    }
    return 0;
}