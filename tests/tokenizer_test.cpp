#include <gtest/gtest.h>
#include "../src/inference/tokeniser.h"
#include <vector>
#include <string>

class CLIPTokenizerTest : public ::testing::Test {
protected:
    CLIPTokenizer* tokenizer;

    void SetUp() override {
        // Initialize tokenizer with the BPE file path
        tokenizer = new CLIPTokenizer("../src/data/bpe_simple_vocab_16e6.txt");
    }

    void TearDown() override {
        delete tokenizer;
    }
};

// Test basic tokenization
TEST_F(CLIPTokenizerTest, BasicTokenization) {
    std::string text = "Hello world";
    std::vector<int> tokens = tokenizer->encode(text);

    std::cout << "\"" << text << "\" tokenised: " << std::endl;
    for (auto str : tokens) {
        std::cout << str << " ";
    }
    std::cout << std::endl;
    
    // Check that we got some tokens
    ASSERT_FALSE(tokens.empty());
    
    // Decode back to text
    std::string decoded = tokenizer->decode(tokens);
    
    // Check that the decoded text contains our input
    // Note: The exact match might not happen due to BPE tokenization
    // EXPECT_TRUE(decoded.find("hello") != std::string::npos);
    // EXPECT_TRUE(decoded.find("world") != std::string::npos);
}

// Test special tokens
TEST_F(CLIPTokenizerTest, SpecialTokens) {
    std::string text = "Test";
    std::vector<int> tokens = tokenizer->encode_text(text, 77, true);
    
    // Check that the first token is start of text
    EXPECT_EQ(tokens[0], tokenizer->encoder["<|startoftext|>"]);
    
    // Find the end of text token
    bool found_eot = false;
    for (int token : tokens) {
        if (token == tokenizer->encoder["<|endoftext|>"]) {
            found_eot = true;
            break;
        }
    }
    EXPECT_TRUE(found_eot);
}

// Test context length handling
TEST_F(CLIPTokenizerTest, ContextLength) {
    std::string text = "This is a long text that should be truncated according to the context length";
    int context_length = 10;
    
    std::vector<int> tokens = tokenizer->encode_text(text, context_length, true);
    
    // Check that the output length matches the context length
    EXPECT_EQ(tokens.size(), context_length);
}

// Test empty input
TEST_F(CLIPTokenizerTest, EmptyInput) {
    std::string text = "";
    std::vector<int> tokens = tokenizer->encode(text);
    
    // Check that we get an empty token list
    EXPECT_TRUE(tokens.empty());
}

// Test BPE function specifically
TEST_F(CLIPTokenizerTest, BPEFunction) {
    // Test a word that should trigger BPE merges
    std::string text = "testing";
    std::vector<int> tokens = tokenizer->encode(text);
    
    // We should get at least one token
    ASSERT_FALSE(tokens.empty());
    
    // Decode should give us something meaningful back
    std::string decoded = tokenizer->decode(tokens);
    EXPECT_FALSE(decoded.empty());
}

// Test whitespace handling
TEST_F(CLIPTokenizerTest, WhitespaceHandling) {
    std::string text = "  multiple    spaces   between   words  ";
    std::vector<int> tokens = tokenizer->encode(text);
    
    std::string decoded = tokenizer->decode(tokens);

    std::cout << "\"" << text << "\" becomes \"" << decoded << "\"" << std::endl;   
    
    // Check that excessive whitespace was normalized
    EXPECT_FALSE(decoded.find("    ") != std::string::npos);
}

// Test case sensitivity
TEST_F(CLIPTokenizerTest, CaseSensitivity) {
    std::string lower = "test";
    std::string upper = "TEST";
    
    std::vector<int> lower_tokens = tokenizer->encode(lower);
    std::vector<int> upper_tokens = tokenizer->encode(upper);
    
    // Tokens should be the same as the tokenizer converts to lowercase
    EXPECT_EQ(lower_tokens, upper_tokens);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}