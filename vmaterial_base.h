#ifndef VULKAN_ENGINE_VMATERIAL_BASE_H
#define VULKAN_ENGINE_VMATERIAL_BASE_H

#include "vmaterial.h"

namespace ve {
    class VMaterialBase : public ve::VMaterial
    {
    public:
        explicit VMaterialBase();
        ~VMaterialBase();

        virtual void InitMaterial();
    };
}

#endif // !VULKAN_ENGINE_VMATERIAL_BASE_H


