#include "FileReader.h"

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>

// #include "../entities/Point.cpp"
#include "../entities/Mbr.h"
using namespace std;


FileReader::FileReader()
{
}

FileReader::FileReader(string filename, string delimeter)
{
    this->filename = filename;
    this->delimeter = delimeter;
}

vector<vector<string>> FileReader::get_data(string path)
{
    ifstream file(path);

    vector<vector<string>> data_list;

    string line = "";
    // Iterate through each line and split the content using delimeter
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        data_list.push_back(vec);
    }
    // Close the File
    file.close();

    return data_list;
}

vector<vector<string>> FileReader::get_data()
{
    return get_data(this->filename);
}

vector<Point> FileReader::get_points()
{
    ifstream file(filename);
    vector<Point> points;
    string line = "";
    //delimeter = "\t";
    //delimeter = ",";
    while (getline(file, line))
    {
        vector<string> vec;
        //boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        boost::split(vec, line, boost::is_any_of(","));
        
        if (vec.size() > 1)
        {
            vector<float> vec_f;
            for (int i = 0; i < vec.size() - 1; i++)
            {
                // if(i == 6 ) break;
                // if(i == 2 ) break;
                if(i == vec.size() - 1) break;
                vec_f.push_back(stod(vec[i]));
            }
            // Point point(stod(vec[0]), stod(vec[1]));
            Point point(vec_f.size(), vec_f);
            points.push_back(point);
        }
    }
    // Close the File
    file.close();

    return points;
}

vector<Mbr> FileReader::get_mbrs()
{
    ifstream file(filename);

    vector<Mbr> mbrs;

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        vector<float> vec_l;
        vector<float> vec_h;
        // for(int i = 0; i < vec.size(); i++)
        // {
        //     if(i < vec.size() / 2)
        //         vec_l.push_back(stod(vec[i]));
        //     else vec_h.push_back(stod(vec[i]));
        // }
        // for(int i = 0; i < 12; i++)
        // {
        //     if(i < 6)
        //         vec_l.push_back(stod(vec[i]));
        //     else vec_h.push_back(stod(vec[i]));
        // }
        // for(int i = 0; i < 4; i++)
        // {
        //     if(i < 2)
        //         vec_l.push_back(stod(vec[i]));
        //     else vec_h.push_back(stod(vec[i]));
        // }
        for(int i = 0; i < vec.size() - 1; i++)
        {
            if(i < (vec.size() - 1) / 2)
                vec_l.push_back(stod(vec[i]));
            else vec_h.push_back(stod(vec[i]));
        }
        // Mbr mbr(stod(vec[0]), stod(vec[1]), stod(vec[2]), stod(vec[3]));
        // int dim = 2;
        int dim = vec_l.size();
        Mbr mbr(dim, vec_l, vec_h);
        mbrs.push_back(mbr);
    }
    
    file.close();

    return mbrs;
}

vector<Point> FileReader::get_points(string filename, string delimeter)
{
    ifstream file(filename);

    vector<Point> points;

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        vector<float> vec_f;
        // for(int i = 0; i < vec.size(); i++)
        // {
        //     vec_f.push_back(stod(vec[i]));
        // }
        // for(int i = 0; i < 6; i++)
        // {
        //     vec_f.push_back(stod(vec[i]));
        // }
        // for(int i = 0; i < 2; i++)
        // {
        //     vec_f.push_back(stod(vec[i]));
        // }
        for (int i = 0; i < vec.size() - 1; i++)
        {
            // if(i == 6 ) break;
            // if(i == 2 ) break;
            if (i == vec.size() - 1)
                break;
            vec_f.push_back(stod(vec[i]));
        }
        // Point point(stod(vec[0]), stod(vec[1]));
        Point point(vec_f.size(), vec_f);
        points.push_back(point);
    }
    // Close the File
    file.close();

    return points;
}

vector<Mbr> FileReader::get_mbrs(string filename, string delimeter)
{
    ifstream file(filename);
    // cout<<"filename: "<<filename<<endl;
    vector<Mbr> mbrs;

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        vector<float> vec_l;
        vector<float> vec_h;
        // for(int i = 0; i < vec.size(); i++)
        // {
        //     if(i < vec.size() / 2)
        //         vec_l.push_back(stod(vec[i]));
        //     else vec_h.push_back(stod(vec[i]));
        // }
        // for(int i = 0; i < 12; i++)
        // {
        //     if(i < 6)
        //         vec_l.push_back(stod(vec[i]));
        //     else vec_h.push_back(stod(vec[i]));
        // }
        // for(int i = 0; i < 4; i++)
        // {
        //     if(i < 2)
        //         vec_l.push_back(stod(vec[i]));
        //     else vec_h.push_back(stod(vec[i]));
        // }
        for(int i = 0; i < vec.size(); i++)
        {
            if(i < (vec.size()) / 2)
                vec_l.push_back(stod(vec[i]));
            else vec_h.push_back(stod(vec[i]));
        }
        // Mbr mbr(stod(vec[0]), stod(vec[1]), stod(vec[2]), stod(vec[3]));
        Mbr mbr(vec_l.size(), vec_l, vec_h);
        mbrs.push_back(mbr);
    }
    
    file.close();

    return mbrs;
}
