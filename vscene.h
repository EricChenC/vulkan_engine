#ifndef VULKAN_ENGINE_VSCENE_H
#define VULKAN_ENGINE_VSCENE_H

#include "vobject.h"
#include "vmodel.h"

namespace ve {
    class VScene : public ve::VObject
    {
    public:
        explicit VScene();
        ~VScene();

        void AddModel(const ve::VModel& model);



    };
}

#endif // !VULKAN_ENGINE_VSCENE_H
