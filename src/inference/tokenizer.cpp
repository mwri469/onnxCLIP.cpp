#include "tokenizer.h"
#include <codecvt>
#include <locale>
#include <cmath>

/*
TODO:   
%   Properly implement byte_decoder, wstring is such a headache
%   C++ ftfy workaround
DONE:
%   bpe() infinite loop
%       -Update 24/01/13: 
%       -bpe() still running loop, changed constructor to init bpe_ranks
%       -Adding max iter limit to exit out
%       -Add better debugging using new _debug attr
%       -Add flag to track valid pairs
*/

/**
 * Header function: initialise required variables and set up bpe
 *
 * @param[in] bpe_path str: path to .txt file containing bpe encodings
 * */
CLIPTokenizer::CLIPTokenizer(const std::string bpe_path) {
    // Default pattern matching tokens
    pat = std::regex(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+)", 
                     std::regex::icase);

    // Byte encoding
    std::unordered_map<int, std::string> byte_encoder = bytes_to_unicode();
    /*
    byte_decoder = {};

    // Converter for UTF-8 to UTF-32
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> utf8_to_utf32;

    for (const auto& [k, v] : byte_encoder) {
        // Convert string to wide string
        std::wstring wide_string = bytes_to_wide(v);

        // Ensure the string is not empty and extract the first code point
        if (!wide_string.empty()) {
            int unicode_code_point = static_cast<int>(wide_string[0]);
            byte_decoder[unicode_code_point] = k;
        }
    } */
    // Load BPE merges (you'll need to implement this part)
    // For now, this is a placeholder
    std::vector<std::pair<std::string, std::string>> merges;
    merges = open_bpe(bpe_path);

    // Initialize bpe_ranks with the merges
    for (size_t i = 0; i < merges.size(); ++i) {
        const auto& pair = merges[i];
        bpe_ranks[pair.first][pair.second] = i;
    }

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

/* Deprecated function that was supposed to convert to unicode
std::wstring CLIPTokenizer::bytes_to_wide(const std::string& input) {
    // Convert a UTF-8 string to a wide string
    try {
        std::wstring wide_string(input.size(), L'\0');
        std::mbstowcs(&wide_string[0], input.c_str(), input.size());
        return wide_string;
    } catch (const std::exception& e) {
        std::cerr << "Error in bytes_to_wide with input: " << input << "\n";
        std::cerr << "Exception: " << e.what() << "\n";
        throw;
    }
}
*/ 

/**
 * Default function to open a byte-pair encoding .txt file.                
 *                                                        
 * @param[in] path str: Path to BPE.txt file                                                                                                              
 * @param[out] merges vector<byte, char>: vector of string pairs containing the BPE
 *                 encondings               
 */
std::vector<std::pair<std::string, std::string>> CLIPTokenizer::open_bpe(std::string path)
{
    // init merges data structure and line
    std::vector <std::pair<std::string,std::string>> merges;
    std::string line;

    // Open file and check opening was successful
    std::ifstream bpe_file(path);
    if (!bpe_file.is_open()) {
        std::cerr << "Error opening vocab file" << std::endl;
        merges.push_back({"", ""});
        return merges;
    }

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

/**
 * Function to convert byte string to unicode
 *
 * @param[out] byte_encoder map<int, str>: byte value->character mapping
 */
std::unordered_map<int, std::string> CLIPTokenizer::bytes_to_unicode() {
    std::unordered_map<int, std::string> byte_encoder;
    
    // Ranges of bytes to encode
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);

    // Go through 
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            byte_encoder[b] = static_cast<char>(256 + n);
            ++n;
        }
    }

    // Link byte as key to character representation as value in map 
    for (int b : bs) {
        byte_encoder[b] = static_cast<char>(b);
    }

    return byte_encoder;
}

/**
 * Create pairs map of two tokens in a word stream
 *
 * @param[in] word vector<str>: word stream
 * 
 * @returns pairs set<pair<str, str>>: map of pair.first -> pair.second tokens
 */
std::set<std::pair<std::string, std::string>> CLIPTokenizer::get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> pairs;
    for (size_t i = 0; i < word.size() - 1; ++i) {
        pairs.insert({word[i], word[i+1]});
    }
    return pairs;
}

std::string CLIPTokenizer::basic_clean(const std::string& text) {
    // Demo function implemented to emultate ftfy python library
    // Note: Full implementation of ftfy.fix_text would require additional library
    // This is a simplified version
    std::string cleaned = text;
    
    // Remove leading/trailing whitespace
    cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r\f\v"));
    cleaned.erase(cleaned.find_last_not_of(" \t\n\r\f\v") + 1);
    
    return cleaned;
}

std::string CLIPTokenizer::whitespace_clean(const std::string& text) {
    // Replace multiple whitespaces with a single space
    std::regex ws_regex(R"(\s+)");
    return std::regex_replace(text, ws_regex, " ");
}

/**
 * Determines if a nested map contains a combination of two keys type<str>  
 *                                               
 * @param[in] mapping map<str, map<str, int>>: Outer-most map to search 
 *                through                                   
 * @param[in] key1 str: key to use for finding outer map                                   
 * @param[in] key2 str: key to use for finding inner map                                   
 * @returns bool: `true` if both keys exist, else `false`                                
 */
bool check_keys(std::unordered_map<std::string, std::unordered_map<std::string, int>> mapping, std::string key1, std::string key2) {
    // Find first key in outer map
    auto it_inner = mapping.find(key1);

    if (it_inner != mapping.end()) {
        // Return wether second key is in inner map  
        return it_inner->second.find(key2) != it_inner->second.end();
    }

    // First key does not exist in outer map
    return false;
}

/**
 * Modification of bpe() function in OpenAI's CLIP module:                  
 * https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L62    
 * Performs the byte-pair encoding to help encode some text.                
 *                                                             
 * @param[in] token str: Tokens to convert to word                                          
 *                                                                
 * @returns word str: Converted word                                                     
 */
std::string CLIPTokenizer::bpe(const std::string& token) {
    // Check cache first, memoization
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
    std::set<std::pair<std::string, std::string>> pairs = get_pairs(word);
    
    if (pairs.empty()) {
        return token + "</w>";
    }

    int iteration = 0;

    while (!pairs.empty() 
            // && iteration < _max_iters
            ) {
        // Set min to maximum possible intiger
        int min = 1 << 31;
        min = min - 1; // Computes 2^31 as an integer
        std::pair<std::string, std::string> best_pair = {"", ""};
        bool found_pair = false;
        
        // Iterate through set and get minimum value
        for (auto it = pairs.begin(); it != pairs.end(); ++it) {
            // Check if element exists
            if (check_keys(bpe_ranks, it->first, it->second)) {
                if (bpe_ranks[it->first][it->second] < min) {
                    // Update min value and best pair if better than existing
                    found_pair = true;
                    min = bpe_ranks[it->first][it->second];
                    best_pair.first = it->first;
                    best_pair.second = it->second;
                }
            } else {
                // pair does not exist
                std::cerr << "Pair : " <<it->first << ", " << it->second << " does not exist in bpe_ranks" << std::endl;
                break;
            }
        }

        if (!found_pair) {
            break;
        }

        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            auto it = std::find(word.begin() + i, word.end(), best_pair.first);
            if (it == word.end()) {
                // Add the remaining part of the word
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }

            size_t j = std::distance(word.begin(), it);
            new_word.insert(new_word.end(), word.begin() + i, word.begin() + j);

            // Check if the bigram is found
            if (j < word.size() - 1 && word[j] == best_pair.first && word[j + 1] == best_pair.second) {
                new_word.push_back(best_pair.first + best_pair.second);
                i = j + 2; // Skip the merged pair
            } else {
                new_word.push_back(word[j]);
                i = j + 1;
            }
        }

        word = new_word;

        // Break if only one symbol remains
        if (word.size() == 1) {
            break;
        }

        // Else, update pairs
        pairs = get_pairs(word);
        iteration++;
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

std::vector<int> CLIPTokenizer::encode(const std::string& text) {
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

std::string CLIPTokenizer::decode(const std::vector<int>& tokens) {
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

std::vector<int> CLIPTokenizer::encode_text(
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
