#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

struct Token {
    size_t id;
    float score;
};

class Tokenizer {
private:
    std::unordered_map<std::string, Token> tokenMap;
    std::vector<std::string> idToToken;
public:
    Tokenizer(const std::string& vocabFile);
    std::vector<std::string> preTokenize(const std::string& text) const;
    std::vector<Token> encodeWord(const std::string& word) const;
    std::vector<Token> tokenize(const std::string& text) const;
    size_t vocab_size() const;
    std::string token_text(size_t id) const;
    std::string decode(const std::vector<size_t>& ids) const;

    ~Tokenizer() = default;
};
