#ifndef VULKAN_ENGINE_VMATERIAL_H
#define VULKAN_ENGINE_VMATERIAL_H

#include "vobject.h"

namespace ve {
    class VMaterial : public ve::VObject
    {
    public:
        explicit VMaterial();
        ~VMaterial();

        virtual void InitMaterial() = 0;

    };
}

#endif // !VULKAN_ENGINE_VMATERIAL_H



