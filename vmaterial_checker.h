#ifndef VULKAN_ENGINE_VMATERIAL_CHECKER_H
#define VULKAN_ENGINE_VMATERIAL_CHECKER_H

#include "vmaterial_base.h"

class VMaterialChecker : public ve::VMaterialBase
{
public:
    explicit VMaterialChecker();
    ~VMaterialChecker();

    virtual void InitMaterial();

};

#endif // !VULKAN_ENGINE_VMATERIAL_CHECKER_H


