//#pragma once
//
//#define GLFW_INCLUDE_VULKAN
//#include <GLFW/glfw3.h>
//
//#define GLM_FORCE_RADIANS
//#define GLM_FORCE_DEPTH_ZERO_TO_ONE
//#define GLM_ENABLE_EXPERIMENTAL
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtx/hash.hpp>
//
//#include <iostream>
//#include <fstream>
//#include <stdexcept>
//#include <algorithm>
//#include <chrono>
//#include <vector>
//#include <cstring>
//#include <array>
//#include <set>
//#include <list>
//#include <unordered_map>
//
//#include "vengine_common.h"
//#include "vobject.h"
//#include "vscene.h"
//
//#include "vulkan/vulkan.hpp"
//
//namespace ve {
//    class VShadow : public VObject {
//    public:
//        explicit VShadow();
//        ~VShadow();
//
//        void InitVulkan();
//
//
//    public:
//        void CreateInstance();
//        void SetupDebugCallback();
//        void CreateSurface();
//        void PickPhysicalDevice();
//        void CreateLogicalDevice();
//        void CreateSwapChain();
//        void CreateCommandPool();
//        void CreateFramebuffers();
//        void CreateSemaphores();
//
//        void RecreateSwapChain();
//
//    private:
//        void CreateDepthVertexBuffer();
//        void CreateDepthIndexBuffer();
//        void CreateDepthUniformBuffer();
//        void CreateDepthDescriptorPool();
//        void CreateDepthDescriptorSet();
//        void CreateDepthCommandBuffer();
//
//        (VkPhysicalDevice device);
//
//
//
//    public:
//        const std::vector<const char*> validationLayers = {
//            "VK_LAYER_LUNARG_standard_validation"
//        };
//
//        const std::vector<const char*> deviceExtensions = {
//            VK_KHR_SWAPCHAIN_EXTENSION_NAME
//        };
//
//        struct QueueFamilyIndices {
//            int graphicsFamily = -1;
//            int presentFamily = -1;
//
//            bool isComplete() {
//                return graphicsFamily >= 0 && presentFamily >= 0;
//            }
//        };
//
//        struct SwapChainSupportDetails {
//            VkSurfaceCapabilitiesKHR capabilities;
//            std::vector<VkSurfaceFormatKHR> formats;
//            std::vector<VkPresentModeKHR> presentModes;
//        };
//
//        struct ShadowVertex {
//            glm::vec3 pos;
//
//            static VkVertexInputBindingDescription getBindingDescription() {
//                VkVertexInputBindingDescription bindingDescription = {};
//                bindingDescription.binding = 0;
//                bindingDescription.stride = sizeof(ShadowVertex);
//                bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
//
//                return bindingDescription;
//            }
//
//            static std::array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions() {
//                std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions = {};
//
//                attributeDescriptions[0].binding = 0;
//                attributeDescriptions[0].location = 0;
//                attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
//                attributeDescriptions[0].offset = offsetof(ShadowVertex, pos);
//
//                return attributeDescriptions;
//            }
//
//            bool operator==(const ShadowVertex& other) const {
//                return pos == other.pos;
//            }
//        };
//
//
//
//
//
//    private:
//        VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
//        VkFormat FindDepthFormat();
//        bool HasStencilComponent(VkFormat format);
//        VkImageView CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
//        void CreateImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
//        void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
//        void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
//        void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
//        VkCommandBuffer BeginSingleTimeCommands();
//        void EndSingleTimeCommands(VkCommandBuffer commandBuffer);
//        void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
//        uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
//        VkShaderModule CreateShaderModule(const std::vector<char>& code);
//        VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
//        VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes);
//        VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
//        SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device);
//        bool IsDeviceSuitable(VkPhysicalDevice device);
//        bool CheckDeviceExtensionSupport(VkPhysicalDevice device);
//        QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device);
//        std::vector<const char*> GetRequiredExtensions();
//        bool CheckValidationLayerSupport();
//        std::vector<char> ReadFile(const std::string& filename);
//
//
//
//    private:
//        GLFWwindow* m_Window;
//
//        VkInstance m_Instance;
//        VkDebugReportCallbackEXT callback;
//        VkSurfaceKHR surface;
//        VkQueue m_GraphicsQueue;
//        VkQueue m_PresentQueue;
//        VkPhysicalDevice m_PhysicalDevice;
//        VkDevice m_Device;
//
//        VkSwapchainKHR m_SwapChain;
//        std::vector<VkImage> m_SwapChainImages;
//        VkFormat m_SwapChainImageFormat;
//        VkExtent2D m_SwapChainExtent;
//        std::vector<VkImageView> m_SwapChainImageViews;
//        std::vector<VkFramebuffer> m_SwapChainFramebuffers;
//
//        VkCommandPool m_CommandPool;
//
//
//    private:
//        const int m_width = 800;
//        const int m_height = 600;
//
//        const bool m_EnableValidationLayers = true;
//
//        const std::string m_model_path = "D:/project/vulkan_engine/media/models/two_object.obj";
//
//
//
//    };
//
//}