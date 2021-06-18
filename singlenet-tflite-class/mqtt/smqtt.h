//
// Created by axv on 2021/6/6.
//

#ifndef SINGLENET_TFLITE_CLASS_MQTT_H
#define SINGLENET_TFLITE_CLASS_MQTT_H


//#include "kputil.h"

// mqtt libs
#include "mqtt/client.h"
#include "mqtt/async_client.h"
#include "mqtt/iasync_client.h"


namespace gj{
    const std::string SERVER_ADDRESS { "tcp://116.62.147.247" };
    const std::string CLIENT_ID { "sync_publish_cpp" };
    const std::string TOPIC_Fall { "fall" };
    const std::string TOPIC_Pic { "pic" };

    const int QOS = 1;



    class sample_mem_persistence : virtual public mqtt::iclient_persistence
    {
        // Whether the store is open
        bool open_;

        // Use an STL map to store shared persistence pointers
        // against string keys.
        std::map<std::string, std::string> store_;

    public:
        sample_mem_persistence() : open_(false) {}

        // "Open" the store
        void open(const std::string& clientId, const std::string& serverURI) override {
            std::cout << "[Opening persistence store for '" << clientId
                      << "' at '" << serverURI << "']" << std::endl;
            open_ = true;
        }

        // Close the persistent store that was previously opened.
        void close() override {
            std::cout << "[Closing persistence store.]" << std::endl;
            open_ = false;
        }

        // Clears persistence, so that it no longer contains any persisted data.
        void clear() override {
            std::cout << "[Clearing persistence store.]" << std::endl;
            store_.clear();
        }

        // Returns whether or not data is persisted using the specified key.
        bool contains_key(const std::string &key) override {
            return store_.find(key) != store_.end();
        }

        // Returns the keys in this persistent data store.
        mqtt::string_collection keys() const override {
            mqtt::string_collection ks;
            for (const auto& k : store_)
                ks.push_back(k.first);
            return ks;
        }

        // Puts the specified data into the persistent store.
        void put(const std::string& key, const std::vector<mqtt::string_view>& bufs) override {
            std::cout << "[Persisting data with key '"
                      << key << "']" << std::endl;
            std::string str;
            for (const auto& b : bufs)
                str.append(b.data(), b.size());	// += b.str();
            store_[key] = std::move(str);
        }

        // Gets the specified data out of the persistent store.
        std::string get(const std::string& key) const override {
            std::cout << "[Searching persistence for key '"
                      << key << "']" << std::endl;
            auto p = store_.find(key);
            if (p == store_.end())
                throw mqtt::persistence_exception();
            std::cout << "[Found persistence data for key '"
                      << key << "']" << std::endl;

            return p->second;
        }

        // Remove the data for the specified key.
        void remove(const std::string &key) override {
            std::cout << "[Persistence removing key '" << key << "']" << std::endl;
            auto p = store_.find(key);
            if (p == store_.end())
                throw mqtt::persistence_exception();
            store_.erase(p);
            std::cout << "[Persistence key removed '" << key << "']" << std::endl;
        }
    };

/////////////////////////////////////////////////////////////////////////////
// Class to receive callbacks

    class user_callback : public virtual mqtt::callback
    {
        void connection_lost(const std::string& cause) override {
            std::cout << "\nConnection lost" << std::endl;
            if (!cause.empty())
                std::cout << "\tcause: " << cause << std::endl;
        }

        void delivery_complete(mqtt::delivery_token_ptr tok) override {
            std::cout << "\n\t[Delivery complete for token: "
                      << (tok ? tok->get_message_id() : -1) << "]" << std::endl;
        }

    public:
    };


};



#endif //SINGLENET_TFLITE_CLASS_MQTT_H
