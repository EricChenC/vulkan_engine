#include "vengine.h"
#include "vmaterial_checker.h"
#include "vscene.h"

#include <string>

void main() {
    VMaterialChecker material_chekcer;
    material_chekcer.InitMaterial();

    ve::VModel model;
    model.LoadModel("");
    model.AddMaterial(material_chekcer);

    ve::VScene scene;
    scene.AddModel(model);

    ve::VEngine engine;
    engine.AddScene(scene);
    engine.Run();
}