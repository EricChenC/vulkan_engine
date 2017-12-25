#ifndef VULKAN_ENGINE_VMODEL_H
#define VULKAN_ENGINE_VMODEL_H

#include "vobject.h"
#include "vmaterial.h"

#include <string>

namespace ve {
    class VModel : public ve::VObject
    {
    public:
        explicit VModel();
        ~VModel();

        virtual void LoadModel(const std::string& model_path);
        virtual void AddMaterial(const ve::VMaterial& material);

    private:

    };

   
}

#endif // !VULKAN_ENGINE_VMODEL_H
