#include <iostream>
#include <vector>
#include <string>
#include "../src/inference/tokenizer.hpp"

bool test_basic_tokenization() {
    std::cout << "=== Running test: BasicTokenization ===" << std::endl;
    CLIPTokenizer tokenizer("../src/data/bpe_simple_vocab_16e6.txt");
    std::string text = "Hello world";
    std::vector<int> tokens = tokenizer.encode(text);

    std::cout << "\"" << text << "\" tokenized: ";
    for (auto token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    if (tokens.empty()) {
        std::cerr << "Error: Tokens are empty." << std::endl;
        return false;
    }

    std::string decoded = tokenizer.decode(tokens);
    std::cout << "Decoded text: \"" << decoded << "\"" << std::endl;
    return true;
}

bool test_special_tokens() {
    std::cout << "=== Running test: SpecialTokens ===" << std::endl;
    CLIPTokenizer tokenizer("../src/data/bpe_simple_vocab_16e6.txt");
    std::string text = "Test";
    std::vector<int> tokens = tokenizer.encode_text(text, 77, true);

    int start_token = tokenizer.encoder.at("<|startoftext|>");
    if (tokens.empty() || tokens[0] != start_token) {
        std::cerr << "Error: First token is not <|startoftext|>." << std::endl;
        return false;
    }

    int end_token = tokenizer.encoder.at("<|endoftext|>");
    bool found_eot = false;
    for (int token : tokens) {
        if (token == end_token) {
            found_eot = true;
            break;
        }
    }

    if (!found_eot) {
        std::cerr << "Error: <|endoftext|> not found in tokens." << std::endl;
        return false;
    }

    std::cout << "Special tokens test passed." << std::endl;
    return true;
}

bool test_context_length() {
    std::cout << "=== Running test: ContextLength ===" << std::endl;
    CLIPTokenizer tokenizer("../src/data/bpe_simple_vocab_16e6.txt");
    std::string text = "This is a long text that should be truncated according to the context length";
    int context_length = 10;
    std::vector<int> tokens = tokenizer.encode_text(text, context_length, true);

    if (tokens.size() != context_length) {
        std::cerr << "Error: Tokens size " << tokens.size() << " != " << context_length << std::endl;
        return false;
    }

    std::cout << "Context length correctly truncated to " << context_length << std::endl;
    return true;
}

bool test_empty_input() {
    std::cout << "=== Running test: EmptyInput ===" << std::endl;
    CLIPTokenizer tokenizer("../src/data/bpe_simple_vocab_16e6.txt");
    std::string text = "";
    std::vector<int> tokens = tokenizer.encode(text);

    if (!tokens.empty()) {
        std::cerr << "Error: Tokens not empty for empty input." << std::endl;
        return false;
    }

    std::cout << "Empty input handled correctly." << std::endl;
    return true;
}

bool test_bpe_function() {
    std::cout << "=== Running test: BPEFunction ===" << std::endl;
    CLIPTokenizer tokenizer("../src/data/bpe_simple_vocab_16e6.txt");
    std::string text = "testing";
    std::vector<int> tokens = tokenizer.encode(text);

    if (tokens.empty()) {
        std::cerr << "Error: No tokens generated for BPE test." << std::endl;
        return false;
    }

    std::string decoded = tokenizer.decode(tokens);
    if (decoded.empty()) {
        std::cerr << "Error: Decoded text is empty." << std::endl;
        return false;
    }

    std::cout << "BPE decoded text: \"" << decoded << "\"" << std::endl;
    return true;
}

bool test_whitespace_handling() {
    std::cout << "=== Running test: WhitespaceHandling ===" << std::endl;
    CLIPTokenizer tokenizer("../src/data/bpe_simple_vocab_16e6.txt");
    std::string text = "  multiple    spaces   between   words  ";
    std::vector<int> tokens = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(tokens);

    std::cout << "Original: \"" << text << "\"" << std::endl;
    std::cout << "Decoded:  \"" << decoded << "\"" << std::endl;

    if (decoded.find("    ") != std::string::npos) {
        std::cerr << "Error: Multiple consecutive spaces in decoded text." << std::endl;
        return false;
    }

    return true;
}

bool test_case_sensitivity() {
    std::cout << "=== Running test: CaseSensitivity ===" << std::endl;
    CLIPTokenizer tokenizer("../src/data/bpe_simple_vocab_16e6.txt");
    std::string lower = "test";
    std::string upper = "TEST";
    std::vector<int> lower_tokens = tokenizer.encode(lower);
    std::vector<int> upper_tokens = tokenizer.encode(upper);

    if (lower_tokens != upper_tokens) {
        std::cerr << "Error: Tokens differ between lowercase and uppercase input." << std::endl;
        return false;
    }

    std::cout << "Case insensitivity verified." << std::endl;
    return true;
}

int main() {
    int passed = 0;
    int failed = 0;

    auto run_test = [&](bool (*test_func)(), const std::string& name) {
        bool result = test_func();
        if (result) {
            std::cout << "+++ PASSED +++\n" << std::endl;
            passed++;
        } else {
            std::cout << "--- FAILED ---\n" << std::endl;
            failed++;
        }
    };

    run_test(test_basic_tokenization, "BasicTokenization");
    run_test(test_special_tokens, "SpecialTokens");
    run_test(test_context_length, "ContextLength");
    run_test(test_empty_input, "EmptyInput");
    run_test(test_bpe_function, "BPEFunction");
    run_test(test_whitespace_handling, "WhitespaceHandling");
    run_test(test_case_sensitivity, "CaseSensitivity");

    std::cout << "Test Summary:" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;

    return failed == 0 ? 0 : 1;
}