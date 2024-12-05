#include "tokeniser.h"
#include <codecvt>
#include <locale>

CLIPTokenizer::CLIPTokenizer(const std::string& bpe_path) {
    // Default pattern matching tokens
    pat = std::regex(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+)", 
                     std::regex::icase);

    // Byte encoding
    auto byte_encoder = bytes_to_unicode();
    byte_decoder = {};
    for (const auto& [k, v] : byte_encoder) {
        byte_decoder[static_cast<int>(v)] = k;
    }

    // Load BPE merges (you'll need to implement this part)
    // For now, this is a placeholder
    std::vector<std::pair<std::string, std::string>> merges;
    std::string path="/home/teknique/clip_onnx_cpp_inference/src/data/bpe_simple_vocab_16e6.txt";
    merges = open_bpe(path);

    // Initialize vocabulary
    std::vector<std::string> vocab;
    for (const auto& v : byte_encoder) {
        vocab.push_back(v.second);
        vocab.push_back(v.second + "</w>");
    }

    for (const auto& merge : merges) {
        vocab.push_back(merge.first + merge.second);
    }

    // Add special tokens
    vocab.push_back("<|startoftext|>");
    vocab.push_back("<|endoftext|>");

    // Create encoder and decoder
    for (size_t i = 0; i < vocab.size(); ++i) {
        encoder[vocab[i]] = i;
        decoder[i] = vocab[i];
    }

    // Initialize cache with special tokens
    cache["<|startoftext|>"] = "<|startoftext|>";
    cache["<|endoftext|>"] = "<|endoftext|>";
}

std::vector<std::pair<std::string, std::string>> CLIPTokenizer::open_bpe(std::string &path)
{
    // Open file and check opening was successful
    std::ifstream bpe_file(path);
    if (!bpe_file.is_open()) {
        std::cerr << "Error opening vocab file" << std::endl;
        return 1;
    }

    // init merges data structure and line
    std::vector <std::pair<std::string,std::string>> merges;
    std::string line;

    // Waste first line
    std::getline(bpe_file, line);

    // Read file line-by-line
    while (std::getline(bpe_file, line)) {
        std::istringstream iss(line);
        std::string first, second;

        // Split line into byte pair encoding
        if (iss >> first >> second) {
            merges.push_back({first, second});
        }
    }

    // Close file, return merges
    bpe_file.close();

    return merges;
}

/*
Function to convert byte string to unicode
*/
std::unordered_map<int, std::string> CLIPTokenizer::bytes_to_unicode() {
    std::unordered_map<int, std::string> byte_encoder;
    
    // Ranges of bytes to encode
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            byte_encoder[b] = static_cast<char>(256 + n);
            ++n;
        }
    }

    for (int b : bs) {
        byte_encoder[b] = static_cast<char>(b);
    }

    return byte_encoder;
}

std::set<std::pair<std::string, std::string>> CLIPTokeniser::get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> pairs;
    for (size_t i = 0; i < word.size() - 1; ++i) {
        pairs.insert({word[i], word[i+1]});
    }
    return pairs;
}

std::string CLIPTokeniser::basic_clean(const std::string& text) {
    // Note: Full implementation of ftfy.fix_text would require additional library
    // This is a simplified version
    std::string cleaned = text;
    
    // Remove leading/trailing whitespace
    cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r\f\v"));
    cleaned.erase(cleaned.find_last_not_of(" \t\n\r\f\v") + 1);
    
    return cleaned;
}

std::string CLIPTokeniser::whitespace_clean(const std::string& text) {
    // Replace multiple whitespaces with a single space
    std::regex ws_regex(R"(\s+)");
    return std::regex_replace(text, ws_regex, " ");
}

std::string CLIPTokeniser::bpe(const std::string& token) {
    // Check cache first
    auto cache_it = cache.find(token);
    if (cache_it != cache.end()) {
        return cache_it->second;
    }

    // Prepare the word
    std::vector<std::string> word;
    for (size_t i = 0; i < token.length() - 1; ++i) {
        word.push_back(std::string(1, token[i]));
    }
    word.push_back(token.substr(token.length() - 1) + "</w>");

    // Get initial pairs
    auto pairs = get_pairs(word);
    
    if (pairs.empty()) {
        return token + "</w>";
    }

    // BPE merge process (simplified)
    while (true) {
        // Find the pair with the lowest rank
        auto best_pair = std::min_element(pairs.begin(), pairs.end(), 
            [this](const auto& a, const auto& b) {
                auto rank_a = bpe_ranks.find(a) != bpe_ranks.end() ? 
                    bpe_ranks[a] : std::numeric_limits<int>::max();
                auto rank_b = bpe_ranks.find(b) != bpe_ranks.end() ? 
                    bpe_ranks[b] : std::numeric_limits<int>::max();
                return rank_a < rank_b;
            });

        if (bpe_ranks.find(*best_pair) == bpe_ranks.end()) {
            break;
        }

        // Merge the best pair
        // (Note: Full implementation would be more complex)
    }

    // Convert word back to string
    std::string result = word[0];
    for (size_t i = 1; i < word.size(); ++i) {
        result += " " + word[i];
    }

    // Cache and return
    cache[token] = result;
    return result;
}

std::vector<int> CLIPTokeniser::encode(const std::string& text) {
    std::vector<int> bpe_tokens;
    
    // Clean and lowercase the text
    std::string cleaned_text = whitespace_clean(basic_clean(text));
    std::transform(cleaned_text.begin(), cleaned_text.end(), cleaned_text.begin(), ::tolower);

    // Tokenize using regex
    std::sregex_iterator it(cleaned_text.begin(), cleaned_text.end(), pat);
    std::sregex_iterator end;

    while (it != end) {
        std::string token = it->str();
        
        // Encode token to bytes
        std::string byte_encoded_token;
        for (unsigned char c : token) {
            // Use byte encoder (simplified)
            byte_encoded_token += static_cast<char>(c);
        }

        // Apply BPE
        std::string bpe_token = bpe(byte_encoded_token);

        // Convert to token IDs
        std::istringstream iss(bpe_token);
        std::string sub_token;
        while (iss >> sub_token) {
            if (encoder.find(sub_token) != encoder.end()) {
                bpe_tokens.push_back(encoder[sub_token]);
            }
        }

        ++it;
    }

    return bpe_tokens;
}

std::string CLIPTokeniser::decode(const std::vector<int>& tokens) {
    // Convert tokens back to text
    std::string text;
    for (int token : tokens) {
        if (decoder.find(token) != decoder.end()) {
            text += decoder[token];
        }
    }

    // Remove "</w>" and handle decoding
    text = std::regex_replace(text, std::regex("</w>"), " ");
    
    return text;
}

std::vector<int> CLIPTokeniser::encode_text(
    const std::string& text, 
    int context_length, 
    bool truncate
) {
    // Get start and end of text tokens
    int sot_token = encoder["<|startoftext|>"];
    int eot_token = encoder["<|endoftext|>"];

    // Encode the text
    std::vector<int> tokens = encode(text);

    // Prepare result vector
    std::vector<int> result(context_length, 0);

    // Add start of text token
    result[0] = sot_token;

    // Add encoded tokens
    size_t max_tokens = std::min(tokens.size(), static_cast<size_t>(context_length - 2));
    for (size_t i = 0; i < max_tokens; ++i) {
        result[i + 1] = tokens[i];
    }

    // Add end of text token
    result[max_tokens + 1] = eot_token;

    return result;
}
