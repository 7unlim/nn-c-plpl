#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <regex>


// for convenience
using json = nlohmann::json;

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    bool in_quotes = false;
    for (char c : s) {
        if (c == '"') {
            in_quotes = !in_quotes;
        }
        if (c == delimiter && !in_quotes) {
            tokens.push_back(token);
            token.clear();
        } else {
            token += c;
        }
    }
    tokens.push_back(token); 
    return tokens;
}

// std::string replace_underscores(const std::string &str) {
//     std::string result = str;
//     std::replace(result.begin(), result.end(), '_', ' ');
//     return result;
// }

void transform_data() {
    std::ifstream ifs("loan_approval_dataset.json");
    std::ofstream ofs("transformed_data.csv");

    json j;
    ifs >> j;

    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> data;

    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() != "Id") {
            headers.push_back(it.key());
        }
    }

    size_t row_count = j[headers[0]].size();
    data.resize(row_count, std::vector<std::string>(headers.size()));

    for (size_t col = 0; col < headers.size(); ++col) {
        std::string header = headers[col];
        for (size_t row = 0; row < row_count; ++row) {
            data[row][col] = j[header][std::to_string(row)].dump();
        }
    }

    for (size_t i = 0; i < headers.size(); ++i) {
        std::string header = headers[i];
        ofs << header;
        if (i < headers.size() - 1) {
            ofs << ","; 
        }
    }
    ofs << std::endl;

    for (const auto &row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            ofs << row[i];
            if (i < row.size() - 1) {
                ofs << ",";
            }
        }
        ofs << std::endl;
    }

    ifs.close();
    ofs.close();
}

std::vector<double> oneHotEncode(const std::string& key, const std::unordered_map<std::string, int>& mapping, int num_categories) {
    std::vector<double> one_hot(num_categories, 0.0);
    auto it = mapping.find(key);
    if (it != mapping.end()) {
        one_hot[it->second] = 1.0;
    }
    return one_hot;
}

double cleanNumericValue(const std::string& value) {
    std::regex re("[^0-9.]"); 
    std::string cleaned = std::regex_replace(value, re, ""); 
    try {
        return std::stod(cleaned); 
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error converting value to double: " << e.what() << std::endl;
        return 0.0; 
    }
}

void normalize(std::vector<double> &column) {
    double max_value = *std::max_element(column.begin(), column.end());
    double min_value = *std::min_element(column.begin(), column.end());
    for (auto &val : column) {
        if (max_value != min_value) {
            val = (val - min_value) / (max_value - min_value);
        } else {
            val = 0.0; 
        }
    }
}

void encode_data(const std::string& file) {
    std::unordered_map<std::string, int> encoded_cities;
    std::unordered_map<std::string, int> encoded_states;
    std::unordered_map<std::string, int> encoded_professions;

    std::vector<std::vector<std::string>> data_;
    std::vector<std::vector<double>> encoded_data;

    int city_counter = 0;
    int state_counter = 0;
    int professions_counter = 0;

    std::ifstream ifs(file);
    std::ofstream ofs("encoded_test_data.csv");

    if (!ifs.is_open()) {
        throw std::runtime_error("Input file failed to open");
    }

    if (!ofs.is_open()) {
        throw std::runtime_error("Output file failed to open");
    }

    std::string line;
    bool header = true;
    while (std::getline(ifs, line)) {
        if (header) {
            header = false;
            continue; 
        }
        data_.push_back(split(line, ','));
    }

    for (const auto &row : data_) {
        if (encoded_cities.find(row[1]) == encoded_cities.end()) {
            encoded_cities[row[1]] = city_counter++;
        }
        if (encoded_professions.find(row[9]) == encoded_professions.end()) {
            encoded_professions[row[9]] = professions_counter++;
        }
        if (encoded_states.find(row[11]) == encoded_states.end()) {
            encoded_states[row[11]] = state_counter++;
        }
    }

    ofs << "Age,";
    for (int i = 0; i < city_counter; ++i) ofs << "CITY_" << i << ",";
    for (int i = 0; i < professions_counter; ++i) ofs << "Profession_" << i << ",";
    for (int i = 0; i < state_counter; ++i) ofs << "STATE_" << i << ",";
    ofs << "House_Ownership,Car_Ownership,Experience,Married/Single,CURRENT_HOUSE_YRS,CURRENT_JOB_YRS,Income,Risk_Flag" << std::endl;

    // Encode data
    for (const auto &row : data_) {
        std::vector<double> processedRow;

        try {
            processedRow.push_back(std::stod(row[0]));

            std::vector<double> city_one_hot(city_counter, 0.0);
            city_one_hot[encoded_cities[row[1]]] = 1.0;
            processedRow.insert(processedRow.end(), city_one_hot.begin(), city_one_hot.end());

            std::vector<double> profession_one_hot(professions_counter, 0.0);
            profession_one_hot[encoded_professions[row[9]]] = 1.0;
            processedRow.insert(processedRow.end(), profession_one_hot.begin(), profession_one_hot.end());

            std::vector<double> state_one_hot(state_counter, 0.0);
            state_one_hot[encoded_states[row[11]]] = 1.0;
            processedRow.insert(processedRow.end(), state_one_hot.begin(), state_one_hot.end());

            processedRow.push_back(row[6] == "\"owned\"" ? 1.0 : 0.0);

            processedRow.push_back(row[4] == "\"yes\"" ? 1.0 : 0.0);

            processedRow.push_back(cleanNumericValue(row[5]));

            processedRow.push_back(row[8] == "\"married\"" ? 1.0 : 0.0);

            for (size_t i = 2; i < row.size(); ++i) {
                if (i == 4 || i == 6 || i == 8 || i == 9 || i == 11) continue;
                if (i == 0 || i == 5) continue; 
                try {
                    processedRow.push_back(std::stod(row[i]));
                } catch (const std::invalid_argument &e) {
                    std::cerr << "Non-numeric value found in row " << row[1] << ": " << row[i] << ", setting to 0.0" << std::endl;
                    processedRow.push_back(0.0);
                }
            }
        } catch (const std::invalid_argument &e) {
            std::cerr << "Error processing row: " << e.what() << std::endl;
            continue;
        }

        encoded_data.push_back(processedRow);
    }

    std::vector<double> age_column;
    for (const auto &row : encoded_data) {
        age_column.push_back(row[0]);
    }
    normalize(age_column); 
    for (size_t row = 0; row < encoded_data.size(); ++row) {
        encoded_data[row][0] = age_column[row];
    }

    for (size_t col = 0; col < encoded_data[0].size(); ++col) {
        bool is_one_hot_column = false;
        if (col == 0) continue; 
        if (col > 0 && col < city_counter + 1) is_one_hot_column = true; 
        if (col >= city_counter + 1 && col < city_counter + professions_counter + 1) is_one_hot_column = true;
        if (col >= city_counter + professions_counter + 1 && col < city_counter + professions_counter + state_counter + 1) is_one_hot_column = true; 
        if (is_one_hot_column) continue;

        std::vector<double> column_data;
        for (const auto &row : encoded_data) {
            column_data.push_back(row[col]);
        }

        normalize(column_data);

        for (size_t row = 0; row < encoded_data.size(); ++row) {
            encoded_data[row][col] = column_data[row];
        }
    }


    for (const auto &row : encoded_data) {
        for (size_t i = 0; i < row.size(); ++i) {
            ofs << row[i];
            if (i != row.size() - 1) {
                ofs << ",";
            }
        }
        ofs << "\n";
    }
    ofs.close();
}