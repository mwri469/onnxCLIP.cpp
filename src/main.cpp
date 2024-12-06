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
#include <tokeniser.h>

int main(int argc, char* argv[]) {
    std::unordered_map<int, char> byte_decoder;
    static CLIPTokenizer tokenizer;

    std::setlocale(LC_ALL, "en_US.UTF-8");

    if (argc != 2) {
        std::cout << "Usage: ./print_vocab <path/to/vocab.txt>" << std::endl;
        return 1;
    }
    std::string path = argv[1];

    std::ifstream bpe_file(path);
    if (!bpe_file.is_open()) {
        std::cerr << "Error opening vocab file" << std::endl;
        return 1;
    }

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

    bpe_file.close();

    std::string text = "a photo of clip";
    std::vector<int> tokens = tokenizer.encode(text);

    for (auto itr = tokens.begin(); itr != tokens.end(); ++itr) {
        std::cout << *itr << " ";
    }
    std::cout << std::endl;

    return 0;
}