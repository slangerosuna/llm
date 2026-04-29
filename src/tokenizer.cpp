#include "tokenizer.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

Tokenizer::Tokenizer(const std::string& vocabFile) {
    std::ifstream f(vocabFile);
    if (!f.is_open())
        throw std::runtime_error("cannot open vocab file: " + vocabFile);

    std::string line;
    std::getline(f, line); // skip header row

    size_t id = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;

        std::string token, score_str;
        if (line.front() == '"') {
            // Quoted field: find closing quote (handle escaped "" pairs)
            size_t pos = 1;
            while (pos < line.size()) {
                if (line[pos] == '"') {
                    if (pos + 1 < line.size() && line[pos + 1] == '"')
                        pos += 2; // skip escaped quote
                    else
                        break;
                } else {
                    ++pos;
                }
            }
            // pos is the closing quote index; unescape
            std::string raw = line.substr(1, pos - 1);
            for (size_t i = 0; i < raw.size(); ++i) {
                if (raw[i] == '"' && i + 1 < raw.size() && raw[i + 1] == '"')
                    { token += '"'; ++i; }
                else
                    token += raw[i];
            }
            score_str = line.substr(pos + 2); // skip closing quote and comma
        } else {
            // Unquoted: last comma separates token from score
            auto comma = line.rfind(',');
            token = line.substr(0, comma);
            score_str = line.substr(comma + 1);
        }

        float score = std::stof(score_str);
        tokenMap[token] = Token{id++, score};
        idToToken.push_back(token);
    }
}

std::vector<std::string> Tokenizer::preTokenize(const std::string& text) const {
    std::vector<std::string> words;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace((unsigned char)text[i])) ++i;
        if (i >= text.size()) break;
        size_t start = i;
        while (i < text.size() && !std::isspace((unsigned char)text[i])) ++i;
        words.push_back("_" + text.substr(start, i - start));
    }
    return words;
}

std::vector<Token> Tokenizer::encodeWord(const std::string& word) const {
    const float kInf = std::numeric_limits<float>::infinity();

    struct Seg { size_t start; float score; };
    std::vector<Seg> best(word.size() + 1, {std::string::npos, kInf});
    best[0] = {0, 0.0f};

    for (size_t start = 0; start < word.size(); ++start) {
        if (best[start].score == kInf) continue;
        float base = best[start].score;
        for (size_t end = start + 1; end <= word.size(); ++end) {
            auto it = tokenMap.find(word.substr(start, end - start));
            if (it == tokenMap.end()) continue;
            float score = base + it->second.score;
            if (score < best[end].score)
                best[end] = {start, score};
        }
    }

    if (best[word.size()].score == kInf) {
        auto unk = tokenMap.find("<unk>");
        return {unk != tokenMap.end() ? unk->second : Token{0, 0.0f}};
    }

    std::vector<Token> tokens;
    size_t end = word.size();
    while (end != 0) {
        size_t start = best[end].start;
        tokens.push_back(tokenMap.at(word.substr(start, end - start)));
        end = start;
    }
    std::reverse(tokens.begin(), tokens.end());
    return tokens;
}

std::vector<Token> Tokenizer::tokenize(const std::string& text) const {
    std::vector<Token> result;
    for (const auto& word : preTokenize(text)) {
        auto toks = encodeWord(word);
        result.insert(result.end(), toks.begin(), toks.end());
    }
    return result;
}

size_t Tokenizer::vocab_size() const {
    return idToToken.size();
}

std::string Tokenizer::token_text(size_t id) const {
    if (id >= idToToken.size()) {
        auto it = tokenMap.find("<unk>");
        if (it != tokenMap.end() && it->second.id < idToToken.size()) {
            return idToToken[it->second.id];
        }
        return "<unk>";
    }
    return idToToken[id];
}

std::string Tokenizer::decode(const std::vector<size_t>& ids) const {
    std::string out;
    for (size_t id : ids) {
        const std::string piece = token_text(id);
        for (char c : piece) {
            if (c == '_') {
                out.push_back(' ');
            } else {
                out.push_back(c);
            }
        }
    }
    if (!out.empty() && out.front() == ' ') {
        out.erase(out.begin());
    }
    return out;
}