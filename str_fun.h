#pragma once
#include <vector>
#include <string>

/**
 * Funkcja zwracaj�ca mapowanie na s�ownik z ci�gu znak�w.
 * 
 * \param line napis z kt�rego b�dzie tworzone mapowanie
 * \return Lista gdzie pierwszy argument to nazwa parametru a drugi to jego warto��
 */
std::vector<std::string> get_key_value(std::string line) {
    std::vector<std::string> maps = std::vector<std::string>();
    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
    auto value = line.find('=');
    maps.push_back(line.substr(0, value));
    maps.push_back(line.substr(value + 1, line.length()));
    return maps;
}

/**
 * Funkcja do rozbijania ci�gu znak�w.
 * 
 * \param s Ci�g znak�w do rozbicia
 * \param c Znak wed�ug ktorego nast�puje rozbicie
 * \return Lista ci�g�w po rozbiciu wed�ug \param c
 */
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