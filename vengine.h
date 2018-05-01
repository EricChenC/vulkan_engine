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

namespace ve {
    class VEngine : public ve::VObject
    {
    public:
        explicit VEngine();
        ~VEngine();

        virtual void Run();
        virtual void AddScene(const ve::VScene& scene);

        const VkDevice get_logic_device();
        void AddUniformBufferAndMemory(const VkBuffer& uniform_buffer, const VkDeviceMemory& uniform_buffer_memory);

    protected:
        virtual void Draw();

        virtual void Clear();

        void CreateTexture(const std::string& texture_path, VkImage& texture, VkDeviceMemory& texture_memory);
        void CreateTextureView(VkImageView& view, VkImage& texture);
        void CreateTextureSampler(VkSampler& sampler);

    private:
        VkDevice logic_device_;

        std::vector<VkBuffer> uniform_buffers_;
        std::vector<VkDeviceMemory> uniform_buffer_memorys_;


    public:
        void initWindow();
        void initVulkan();
        void mainLoop();
        void cleanupSwapChain();
        void cleanup();
        void recreateSwapChain();
        void createInstance();
        void setupDebugCallback();
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

        void SetTextureInfo();

        void loadModel();
        void loadShadowModel();
        void createVertexBuffer();
        void createIndexBuffer();
        void createUniformBuffer();
        void createDescriptorPool();
        void createDescriptorSet();
        void CreateCommandBuffers();

        void createSemaphores();
        void updateUniformBuffer();
        void camera_control();
        void drawFrame();

        void RecreateBufer();


        /* shadow */
        void CreateShadowFrameBuffer();
        void CreateShadowRenderPass();
        void CreateShadowLayout();
        void CreateShadowVertexBuffer();
        void CreateShadowIndexBuffer();
        void CreateShadowUniformBuffer();
        void CreateShadowDescriptorPool();
        void CreateShadowDescriptorSet();
        void CreateShadowPipeline();
        void CreateShadowCommandBuffer();

        void UpdateShadowUniformBuffer();

    public:
        struct UniformMatrixBufferObject {
            glm::mat4 view;
            glm::mat4 proj;
            glm::mat4 lightSpace;
            glm::vec3 lightPos;
        };

        struct ConstantMatrixModel {
            glm::mat4 model;
        };

        // normal 
        struct UniformNormalParameters{
            glm::vec4 ambientColor = glm::vec4(0.2f, 0.2f, 0.2f, 0.2f);						// ambient color
            glm::vec4 diffuseColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);						// diffuse color
            glm::vec4 specularColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);					// specular color
            glm::vec4 transparency = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);						// transparency

            float diffuseRough = 1.0f;						// diffuse roughness
            float shininess = 1.0f;						// specular shininess
            float reflectivity = 1.0f;						// specular reflectivity
            float indexOfRefraction = 1.0f;				// index of refraction
            float extinction = 1.0f;						// extinction of metal
            float opacity = 1.0f;

            unsigned int  options = 1;
            unsigned int  version = 2;							// shader version
        };

        struct UniformNormalTextureParameters {
            glm::vec2 diffuseOffset = glm::vec2(0.0f, 0.0f);		// UV pixel offset
            glm::vec2 diffuseRepeat = glm::vec2(1.0f, 1.0f);		// UV pixel repeat

            glm::vec2 specularOffset = glm::vec2(0.0f, 0.0f);		// UV pixel offset
            glm::vec2 specularRepeat = glm::vec2(1.0f, 1.0f);		// UV pixel repeat

            glm::vec2 bumpOffset = glm::vec2(0.0f, 0.0f);			// UV pixel offset
            glm::vec2 bumpRepeat = glm::vec2(1.0f, 1.0f);			// UV pixel repeat

            float diffuseScale = 1.0f;		// diffuse scale
            float specularScale = 1.0f;		// specular scale
            float bumpScale = 1.0f;			// bump scale
        };

        struct UniformSpecialParameters {
            glm::vec4 tintColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

            float shininessU = 0.4f;				// specular shininess
            float shininessV = 0.4f;				// specular shininess
            float cutOff = 1.1f;					// cut-off opacity
        };

        struct UniformSpecialTextureParameters {
            glm::vec2 cutOffOffset = glm::vec2(0.0f, 0.0f);		// UV pixel offset
            glm::vec2 cutOffRepeat = glm::vec2(1.0f, 1.0f);		// UV pixel repeat
            float cutOffScale = 0.5f;		// default scale
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


        struct ShadowUBO {
            glm::mat4 depthMVP;
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

        // diffuse texture
        VkImage diffuseTexture;
        VkDeviceMemory diffuseTextureMemory;
        VkImageView diffuseTextureView;
        VkSampler diffuseTextureSampler;

        // specular texture
        VkImage specularTexture;
        VkDeviceMemory specularTextureMemory;
        VkImageView specularTextureView;
        VkSampler specularTextureSampler;

        // bump texture
        VkImage bumpTexture;
        VkDeviceMemory bumpTextureMemory;
        VkImageView bumpTextureView;
        VkSampler bumpTextureSampler;

        // cutoff texture
        VkImage cutoffTexture;
        VkDeviceMemory cutoffTextureMemory;
        VkImageView cutoffTextureView;
        VkSampler cutoffTextureSampler;

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkBuffer indexBuffer;
        VkDeviceMemory indexBufferMemory;

        // uniform vp buffer
        VkBuffer uniformMatrixBuffer;
        VkDeviceMemory uniformMatrixBufferMemory;

        // uniform normal buffer
        VkBuffer uniformNormalBuffer;
        VkDeviceMemory uniformNormalBufferMemory;

        //uniform normal texture buffer
        VkBuffer uniformNormalTextureBuffer;
        VkDeviceMemory uniformNormalTextureBufferMemory;

        //uniform special buffer
        VkBuffer uniformSpecialBuffer;
        VkDeviceMemory uniformSpecialBufferMemory;

        //uniform special texture buffer
        VkBuffer uniformSpecialTextureBuffer;
        VkDeviceMemory uniformSpecialTextureBufferMemory;

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

        VkFramebuffer shadowFramebuffers;
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
        VkDescriptorSet shadowDescriptorSet;
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

        // Initial position : on +Z
        glm::vec3 position = glm::vec3(0, 2, 10);
        // Initial horizontal angle : toward -Z
        float horizontalAngle = 3.14f;
        // Initial vertical angle : none
        float verticalAngle = 0.0f;
        // Initial Field of View
        float initialFoV = 45.0f;

        float speed = 3.0f; // 3 units / second
        float mouseSpeed = 0.005f;

        double last_xpos_ = 0.0f;
        double last_ypos_ = 0.0f;

        ShadowUBO ubo = {};

        glm::vec3 lightPos = glm::vec3(10.1f, -10.0f, 0.1f);

        uint32_t shadow_width = 2048;
        uint32_t shadow_height = 2048;

        bool kFirstPress = true;

        const bool enableValidationLayers = true;

        // version 2
        const std::string MODEL_PATH = "D:/project/vulkan_engine/media/models/shadow.obj";
        const std::string SHADOW_MODEL_PATH = "D:/project/vulkan_engine/media/models/shadow.obj";
        const std::string TEXTURE_PATH = "D:/project/vulkan_engine/media/revite_textures/Masonry.Stone.Limestone.Rustic.png";
        const std::string SPECULAR_TEXTURE_PATH = "D:/project/vulkan_engine/media/revite_textures/Masonry.Stone.Limestone.Rustic.bump.png";
        const std::string BUMP_TEXTURE_PATH = "D:/project/vulkan_engine/media/revite_textures/Masonry.Stone.Limestone.Rustic.bump-normal.png";
        const std::string CUTOFF_TEXTURE_PATH = "D:/project/vulkan_engine/media/revite_textures/Metal-cutoff02.png";

       
    };
}

#endif // !VULKAN_ENGINE_VENGINE_H

