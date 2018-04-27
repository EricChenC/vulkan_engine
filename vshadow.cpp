//#include "vshadow.h"
//
//#define STB_IMAGE_IMPLEMENTATION
//#include <stb_image.h>
//
//#define TINYOBJLOADER_IMPLEMENTATION
//#include <tiny_obj_loader.h>
//
//
//namespace std {
//    template<> struct hash<ve::VShadow::ShadowVertex> {
//        size_t operator()(ve::VShadow::ShadowVertex const& vertex) const {
//            return (hash<glm::vec3>()(vertex.pos));
//        }
//    };
//}
//
//VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
//    auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
//    if (func != nullptr) {
//        return func(instance, pCreateInfo, pAllocator, pCallback);
//    }
//    else {
//        return VK_ERROR_EXTENSION_NOT_PRESENT;
//    }
//}
//
//void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
//    auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
//    if (func != nullptr) {
//        func(instance, callback, pAllocator);
//    }
//}
//
//static void onWindowResized(GLFWwindow* window, int width, int height) {
//    if (width == 0 || height == 0) return;
//
//    ve::VShadow* app = reinterpret_cast<ve::VShadow*>(glfwGetWindowUserPointer(window));
//    app->RecreateSwapChain();
//}
//
//static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t obj, size_t location, int32_t code, const char* layerPrefix, const char* msg, void* userData) {
//    std::cerr << "validation layer: " << msg << std::endl;
//
//    return VK_FALSE;
//}
//
//
//
//ve::VShadow::VShadow()
//{
//}
//
//ve::VShadow::~VShadow()
//{
//}
//
//void ve::VShadow::InitVulkan()
//{
//}
//
//void ve::VShadow::CreateInstance()
//{
//    if (m_EnableValidationLayers && !CheckValidationLayerSupport()) {
//        throw std::runtime_error("validation layers requested, but not available!");
//    }
//
//    VkApplicationInfo appInfo = {};
//    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
//    appInfo.pApplicationName = "Hello Shadow";
//    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
//    appInfo.pEngineName = "No Engine";
//    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
//    appInfo.apiVersion = VK_API_VERSION_1_0;
//
//    VkInstanceCreateInfo createInfo = {};
//    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
//    createInfo.pApplicationInfo = &appInfo;
//
//    auto extensions = GetRequiredExtensions();
//    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
//    createInfo.ppEnabledExtensionNames = extensions.data();
//
//    if (m_EnableValidationLayers) {
//        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
//        createInfo.ppEnabledLayerNames = validationLayers.data();
//    }
//    else {
//        createInfo.enabledLayerCount = 0;
//    }
//
//    if (vkCreateInstance(&createInfo, nullptr, &m_Instance) != VK_SUCCESS) {
//        throw std::runtime_error("failed to create instance!");
//    }
//}
//
//void ve::VShadow::SetupDebugCallback()
//{
//    if (!m_EnableValidationLayers) return;
//
//    VkDebugReportCallbackCreateInfoEXT createInfo = {};
//    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
//    createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
//    createInfo.pfnCallback = debugCallback;
//
//    if (CreateDebugReportCallbackEXT(m_Instance, &createInfo, nullptr, &callback) != VK_SUCCESS) {
//        throw std::runtime_error("failed to set up debug callback!");
//    }
//}
//
//void ve::VShadow::CreateSurface()
//{
//    if (glfwCreateWindowSurface(m_Instance, m_Window, nullptr, &surface) != VK_SUCCESS) {
//        throw std::runtime_error("failed to create window surface!");
//    }
//}
//
//void ve::VShadow::PickPhysicalDevice()
//{
//    uint32_t deviceCount = 0;
//    vkEnumeratePhysicalDevices(m_Instance, &deviceCount, nullptr);
//
//    if (deviceCount == 0) {
//        throw std::runtime_error("failed to find GPUs with Vulkan support!");
//    }
//
//    std::vector<VkPhysicalDevice> devices(deviceCount);
//    vkEnumeratePhysicalDevices(m_Instance, &deviceCount, devices.data());
//
//    for (const auto& device : devices) {
//        if (IsDeviceSuitable(device)) {
//            m_PhysicalDevice = device;
//            break;
//        }
//    }
//
//    if (m_PhysicalDevice == VK_NULL_HANDLE) {
//        throw std::runtime_error("failed to find a suitable GPU!");
//    }
//}
//
//void ve::VShadow::CreateLogicalDevice()
//{
//    QueueFamilyIndices indices = FindQueueFamilies(m_PhysicalDevice);
//
//    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
//    std::set<int> uniqueQueueFamilies = { indices.graphicsFamily, indices.presentFamily };
//
//    float queuePriority = 1.0f;
//    for (int queueFamily : uniqueQueueFamilies) {
//        VkDeviceQueueCreateInfo queueCreateInfo = {};
//        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
//        queueCreateInfo.queueFamilyIndex = queueFamily;
//        queueCreateInfo.queueCount = 1;
//        queueCreateInfo.pQueuePriorities = &queuePriority;
//        queueCreateInfos.push_back(queueCreateInfo);
//    }
//
//    VkPhysicalDeviceFeatures deviceFeatures = {};
//    deviceFeatures.samplerAnisotropy = VK_TRUE;
//
//    VkDeviceCreateInfo createInfo = {};
//    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
//
//    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
//    createInfo.pQueueCreateInfos = queueCreateInfos.data();
//
//    createInfo.pEnabledFeatures = &deviceFeatures;
//
//    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
//    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
//
//    if (m_EnableValidationLayers) {
//        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
//        createInfo.ppEnabledLayerNames = validationLayers.data();
//    }
//    else {
//        createInfo.enabledLayerCount = 0;
//    }
//
//    if (vkCreateDevice(m_PhysicalDevice, &createInfo, nullptr, &m_Device) != VK_SUCCESS) {
//        throw std::runtime_error("failed to create logical device!");
//    }
//
//    vkGetDeviceQueue(m_Device, indices.graphicsFamily, 0, &m_GraphicsQueue);
//    vkGetDeviceQueue(m_Device, indices.presentFamily, 0, &m_PresentQueue);
//}
//
//void ve::VShadow::CreateSwapChain()
//{
//    SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(m_PhysicalDevice);
//
//    VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.formats);
//    VkPresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.presentModes);
//    VkExtent2D extent = ChooseSwapExtent(swapChainSupport.capabilities);
//
//    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
//    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
//        imageCount = swapChainSupport.capabilities.maxImageCount;
//    }
//
//    VkSwapchainCreateInfoKHR createInfo = {};
//    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
//    createInfo.surface = surface;
//
//    createInfo.minImageCount = imageCount;
//    createInfo.imageFormat = surfaceFormat.format;
//    createInfo.imageColorSpace = surfaceFormat.colorSpace;
//    createInfo.imageExtent = extent;
//    createInfo.imageArrayLayers = 1;
//    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
//
//    QueueFamilyIndices indices = FindQueueFamilies(m_PhysicalDevice);
//    uint32_t queueFamilyIndices[] = { (uint32_t)indices.graphicsFamily, (uint32_t)indices.presentFamily };
//
//    if (indices.graphicsFamily != indices.presentFamily) {
//        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
//        createInfo.queueFamilyIndexCount = 2;
//        createInfo.pQueueFamilyIndices = queueFamilyIndices;
//    }
//    else {
//        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
//    }
//
//    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
//    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
//    createInfo.presentMode = presentMode;
//    createInfo.clipped = VK_TRUE;
//
//    if (vkCreateSwapchainKHR(m_Device, &createInfo, nullptr, &m_SwapChain) != VK_SUCCESS) {
//        throw std::runtime_error("failed to create swap chain!");
//    }
//
//    vkGetSwapchainImagesKHR(m_Device, m_SwapChain, &imageCount, nullptr);
//    m_SwapChainImages.resize(imageCount);
//    vkGetSwapchainImagesKHR(m_Device, m_SwapChain, &imageCount, m_SwapChainImages.data());
//
//    m_SwapChainImageFormat = surfaceFormat.format;
//    m_SwapChainExtent = extent;
//
//}
//
//void ve::VShadow::CreateCommandPool()
//{
//    QueueFamilyIndices queueFamilyIndices = FindQueueFamilies(m_PhysicalDevice);
//
//    VkCommandPoolCreateInfo poolInfo = {};
//    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
//    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;
//
//    if (vkCreateCommandPool(m_Device, &poolInfo, nullptr, &m_CommandPool) != VK_SUCCESS) {
//        throw std::runtime_error("failed to create graphics command pool!");
//    }
//}
//
//void ve::VShadow::CreateFramebuffers()
//{
//
//}
//
//void ve::VShadow::CreateSemaphores()
//{
//}
//
//void ve::VShadow::CreateShadowVertexBuffer()
//{
//}
//
//void ve::VShadow::CreateShadowIndexBuffer()
//{
//}
//
//void ve::VShadow::CreateShadowUniformBuffer()
//{
//}
//
//void ve::VShadow::CreateShadowDescriptorPool()
//{
//}
//
//void ve::VShadow::CreateShadowDescriptorSet()
//{
//}
//
//void ve::VShadow::CreateShadowCommandBuffer()
//{
//}
