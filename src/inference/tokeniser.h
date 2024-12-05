#ifndef CLIP_TOKENIZER_H
#define CLIP_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <regex>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iterator>

class CLIPTokenizer {
public:
    CLIPTokenizer(const std::string& bpe_path = "");
    
    // Main encoding methods
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
    
    // Encode text with context length and truncation
    std::vector<int> encode_text(
        const std::string& text, 
        int context_length = 77, 
        bool truncate = false
    );

private:
    // Byte encoding helpers
    std::unordered_map<int, char> byte_decoder;
    std::unordered_map<int, std::string> decoder;
    std::unordered_map<std::string, int> encoder;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> bpe_ranks;
    
    // Cache for BPE results
    std::unordered_map<std::string, std::string> cache;
    
    // Regex pattern for tokenization
    std::regex pat;

    // Internal helper methods
    std::unordered_map<int, std::string> bytes_to_unicode();
    std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word);
    std::string bpe(const std::string& token);
    std::string basic_clean(const std::string& text);
    std::string whitespace_clean(const std::string& text);
    std::vector<std::pair<std::string, std::string>> open_bpe(std::string& path);

    // Hash function for pair hashing
    struct PairHash {
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1, T2>& p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ h2;
        }
    };
};

#endif // CLIP_TOKENIZER_H