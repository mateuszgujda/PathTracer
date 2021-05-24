#pragma once
#include <vector>
#include <string>

std::vector<std::string> get_key_value(std::string line) {
    std::vector<std::string> maps = std::vector<std::string>();
    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
    auto value = line.find('=');
    maps.push_back(line.substr(0, value));
    maps.push_back(line.substr(value + 1, line.length()));
    return maps;
}


const std::vector<std::string> explode(const std::string& s, const char& c)
{
    std::string buff{ "" };
    std::vector<std::string> v;

    for (auto n : s)
    {
        if (n != c) buff += n; else
            if (n == c && buff != "") { v.push_back(buff); buff = ""; }
    }
    if (buff != "") v.push_back(buff);

    return v;
}