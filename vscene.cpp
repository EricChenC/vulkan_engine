#include "vscene.h"

#include <iostream>

namespace ve {
    VScene::VScene()
    {
    }

    ve::VScene::~VScene()
    {
    }
    void VScene::AddModel(const ve::VModel & model)
    {
        std::cout << "add a model to scene!\n";
    }
}




