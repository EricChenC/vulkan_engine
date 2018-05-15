#ifndef VULKAN_ENGINE_VENGINE_H
#define VULKAN_ENGINE_VENGINE_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <array>
#include <set>
#include <list>
#include <unordered_map>

#include "vengine_common.h"
#include "vobject.h"
#include "vscene.h"

#include "vulkan/vulkan.hpp"

#define SHADOW_MAP_CASCADE_COUNT 4

namespace ve {
    class VEngine : public ve::VObject
    {
    public:
        explicit VEngine();
        ~VEngine();

        void Run();

    protected:
        void CreateTexture(const std::string& texture_path, VkImage& texture, VkDeviceMemory& texture_memory);
        void CreateTextureView(VkImageView& view, VkImage& texture);
        void CreateTextureSampler(VkSampler& sampler);

    private:
        VkDevice logic_device_;

    public:
        void initWindow();
        void initVulkan();
        void mainLoop();
        void cleanupSwapChain();
        void cleanup();
        void recreateSwapChain();
        void createInstance();
        void createSurface();
        void pickPhysicalDevice();
        void createLogicalDevice();
        void createSwapChain();
        void createImageViews();
        void createRenderPass();
        void createDescriptorSetLayout();
        void createGraphicsPipeline();
        void createFramebuffers();
        void createCommandPool();

        void loadModel();
        void createVertexBuffer();
        void createIndexBuffer();
        void createUniformBuffer();
        void createDescriptorPool();
        void createDescriptorSet();
        void CreateCommandBuffers();

        void createSemaphores();
        void updateUniformBuffer();
        void updateSceneUniformBuffer();
        void camera_control();
        void drawFrame();

        void RecreateBufer();


        /* Depth */
        void CreateDepthRenderPass();
        void CreateDepthLayout();
        void CreateDepthVertexBuffer();
        void CreateDepthIndexBuffer();
        void CreateDepthUniformBuffer();
        void CreateDepthDescriptorPool();
        void CreateDepthDescriptorSet();
        void CreateDepthPipeline();
        void CreateDepthCommandBuffer();

        void UpdateDepthUniformBuffer();

    public:
        struct UniformMatrixBufferObject {
            glm::mat4 view;
            glm::mat4 proj;
        };

        struct ConstantMatrixModel {
            glm::mat4 model;
        };

        struct CSM {
            float cascadeSplits[4];
            glm::mat4 cascadeViewProjMat[4];
            glm::vec3 lightDir;
        };


        // For simplicity all pipelines use the same push constant block layout
        struct ShadowPushConstBlock {
            uint32_t cascadeIndex;
        };

        struct ShadowUniformBlock {
            std::array<glm::mat4, SHADOW_MAP_CASCADE_COUNT> cascadeViewProjMat;
        };


        struct QueueFamilyIndices {
            int graphicsFamily = -1;
            int presentFamily = -1;

            bool isComplete() {
                return graphicsFamily >= 0 && presentFamily >= 0;
            }
        };

        struct SwapChainSupportDetails {
            VkSurfaceCapabilitiesKHR capabilities;
            std::vector<VkSurfaceFormatKHR> formats;
            std::vector<VkPresentModeKHR> presentModes;
        };
       
        struct ShadowVertex {
            glm::vec3 pos;

            static VkVertexInputBindingDescription getBindingDescription() {
                VkVertexInputBindingDescription bindingDescription = {};
                bindingDescription.binding = 0;
                bindingDescription.stride = sizeof(ShadowVertex);
                bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

                return bindingDescription;
            }

            static std::array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions() {
                std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions = {};

                attributeDescriptions[0].binding = 0;
                attributeDescriptions[0].location = 0;
                attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
                attributeDescriptions[0].offset = offsetof(ShadowVertex, pos);

                return attributeDescriptions;
            }

            bool operator==(const ShadowVertex& other) const {
                return pos == other.pos;
            }
        };

        struct Vertex {
            glm::vec3 pos;
            glm::vec3 normal;
            glm::vec2 texCoord;

            static VkVertexInputBindingDescription getBindingDescription() {
                VkVertexInputBindingDescription bindingDescription = {};
                bindingDescription.binding = 0;
                bindingDescription.stride = sizeof(Vertex);
                bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

                return bindingDescription;
            }

            static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
                std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

                attributeDescriptions[0].binding = 0;
                attributeDescriptions[0].location = 0;
                attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
                attributeDescriptions[0].offset = offsetof(Vertex, pos);

                attributeDescriptions[1].binding = 0;
                attributeDescriptions[1].location = 1;
                attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
                attributeDescriptions[1].offset = offsetof(Vertex, normal);

                attributeDescriptions[2].binding = 0;
                attributeDescriptions[2].location = 2;
                attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
                attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

                return attributeDescriptions;
            }

            bool operator==(const Vertex& other) const {
                return pos == other.pos && normal == other.normal && texCoord == other.texCoord;
            }
        };

    private:
        VkBool32 findDepthFormat(VkPhysicalDevice physicalDevice, VkFormat *depthFormat);
        bool hasStencilComponent(VkFormat format);
        VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
        void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
        void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
        void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
        VkCommandBuffer beginSingleTimeCommands();
        void endSingleTimeCommands(VkCommandBuffer commandBuffer);
        void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
        VkShaderModule createShaderModule(const std::vector<char>& code);
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes);
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
        bool isDeviceSuitable(VkPhysicalDevice device);
        bool checkDeviceExtensionSupport(VkPhysicalDevice device);
        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
        std::vector<const char*> getRequiredExtensions();
        bool checkValidationLayerSupport();
        std::vector<char> readFile(const std::string& filename);

        void renderScene(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VkDescriptorSet descriptorSet, uint32_t cascadeIndex = 0);


    private:
        GLFWwindow* window;

        VkInstance instance;
        VkDebugReportCallbackEXT callback;
        VkSurfaceKHR surface;

        VkFormat depthFormat;

        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device;

        //VkQueue graphicsQueue;
        VkQueue presentQueue;

        VkSwapchainKHR swapChain;
        std::vector<VkImage> swapChainImages;
        VkFormat swapChainImageFormat;
        VkExtent2D swapChainExtent;
        std::vector<VkImageView> swapChainImageViews;
        std::vector<VkFramebuffer> swapChainFramebuffers;

        VkRenderPass renderPass;
        VkDescriptorSetLayout descriptorSetLayout;
        VkPipelineLayout pipelineLayout;
        VkPipeline graphicsPipeline;

        VkCommandPool commandPool;

        VkImage depthImage;
        VkDeviceMemory depthImageMemory;
        VkImageView depthImageView;

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkBuffer indexBuffer;
        VkDeviceMemory indexBufferMemory;

        // uniform vp buffer
        VkBuffer uniformMatrixBuffer;
        VkDeviceMemory uniformMatrixBufferMemory;

        VkBuffer csmBuffer;
        VkDeviceMemory csmBufferMemory;

        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet;

        std::vector<VkCommandBuffer> commandBuffers;

        VkSemaphore imageAvailableSemaphore;
        VkSemaphore renderFinishedSemaphore;

        struct TextureInfo {
            std::string texture_path;
            VkImage *texture;
            VkDeviceMemory *texture_memory;
            VkImageView *texture_view;
            VkSampler *sampler;
        };

        std::vector<TextureInfo> texture_infos_;


        /* shadow */
        std::vector<ShadowVertex> shadowVertices;
        std::vector<uint32_t> shadowIndices;

        // Contains all resources required for a single shadow map cascade
        struct Cascade {
            VkFramebuffer frameBuffer;
            VkDescriptorSet descriptorSet;
            VkImageView view;

            float splitDepth;
            glm::mat4 viewProjMatrix;

            void destroy(VkDevice device) {
                vkDestroyImageView(device, view, nullptr);
                vkDestroyFramebuffer(device, frameBuffer, nullptr);
            }
        };

        VkRenderPass shadowRenderPass;
        VkImage shadowImage;
        VkDeviceMemory shadowImageMemory;
        VkImageView shadowImageView;
        VkSampler shadowImageSampler;

        VkSemaphore shadowSemaphore;

        VkPipeline shadowPipeline;
        VkPipelineLayout shadowPipelineLayout;

        VkCommandBuffer shadowCommandbuffer;
        
        VkDescriptorPool shadowDescriptorPool;
        VkDescriptorSetLayout shadowDescriptorSetLayout;

        VkBuffer shadowVertexBuffer;
        VkDeviceMemory shadowVertexBufferMemory;

        VkBuffer shadowVertexStagingBuffer;
        VkDeviceMemory shadowVertexStagingBufferMemory;

        VkBuffer shadowIndexStagingBuffer;
        VkDeviceMemory shadowIndexStagingBufferMemory;


        VkBuffer shadowIndexBuffer;
        VkDeviceMemory shadowIndexBufferMemory;
        VkBuffer shadowUniformBuffer;
        VkDeviceMemory shadowUniformBufferMemory;


    private:
        const std::vector<const char*> validationLayers = {
            "VK_LAYER_LUNARG_standard_validation"
        };

        const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };


    private:
        const int WIDTH = 800;
        const int HEIGHT = 600;

        float camera_near_clip_ = 0.1f;
        float camera_far_clip_ = 1000.0f;
        glm::mat4 camera_perspective_;
        glm::mat4 cmare_view_;

        float cascadeSplitLambda = 0.95f;


        // Initial position : on +Z
        glm::vec3 position = glm::vec3(0.01, 2, 10.01);
        // Initial horizontal angle : toward -Z
        float horizontalAngle = 3.14f;
        // Initial vertical angle : none
        float verticalAngle = 0.0f;
        // Initial Field of View
        float initialFoV = 45.0f;

        float speed = 10.0f; // 3 units / second
        float mouseSpeed = 0.005f;

        double last_xpos_ = 0.0f;
        double last_ypos_ = 0.0f;

        ShadowPushConstBlock spcb = {};

        ShadowUniformBlock sub = {};

        //glm::vec3 lightPos = glm::vec3(10.1f, 10.0f, 10.1f);

        // light direction 
        glm::vec3 lightPos = glm::vec3(1.0f, 1.0f, 0.001f);


        glm::mat4 bias = glm::mat4{ 
            0.5, 0.0, 0.0, 0.0,
            0.0, 0.5, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.5, 0.5, 0.0, 1.0 };

        glm::mat4 clip = glm::mat4{
            1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0 };

        uint32_t shadow_width = 4096;
        uint32_t shadow_height = 4096;

        
        std::array<Cascade, SHADOW_MAP_CASCADE_COUNT> cascades;

        bool kFirstPress = true;

        const bool enableValidationLayers = true;

        // version 2
        const std::string MODEL_PATH = "D:/project/vulkan_engine/media/models/shadow.obj";
       
    };
}

#endif // !VULKAN_ENGINE_VENGINE_H

