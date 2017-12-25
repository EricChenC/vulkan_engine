#include "vmodel.h"

#include <iostream>

namespace ve {
    VModel::VModel()
    {
    }

    VModel::~VModel()
    {
    }

    void VModel::LoadModel(const std::string & model_path)
    {
        std::cout << "load a model!\n";
    }

    void VModel::AddMaterial(const ve::VMaterial & material)
    {
        std::cout << "add a material to model\n";
    }

}


