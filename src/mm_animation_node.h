#ifndef MM_ANIMATION_NODE_H
#define MM_ANIMATION_NODE_H

#include "common.h"
#include "mm_animation_library.h"

#include <godot_cpp/classes/animation_root_node.hpp>

#include <queue>

// Godot 4.7 requires the AnimationTree root to be an AnimationRootNode. The plugin
// previously derived from AnimationNodeExtension (a plain AnimationNode), which the
// engine silently rejects as tree_root -> tree_root stays null -> no clip ever plays
// -> the skeleton is never posed by Motion Matching. Switch to AnimationRootNode and
// drive the matching from AnimationNode::_process (its shared per-frame hook).
class MMAnimationNode : public AnimationRootNode {
    GDCLASS(MMAnimationNode, AnimationRootNode);

public:
    GETSET(StringName, library);
    GETSET(real_t, query_frequency, 2.0f)
    GETSET(real_t, transition_halflife, 0.1f)
    GETSET(float, velocity_change_threshold, 50.0f)
    GETSET(float, facing_change_threshold, 0.5f)

    bool blending_enabled{true};
    bool get_blending_enabled() const;

    void set_blending_enabled(bool value);

    // Engine drives us as the AnimationTree root each frame. Return the blend time.
    virtual double _process(double p_time, bool p_seek, bool p_is_external_seeking, bool p_test_only) override;
    virtual Array _get_parameter_list() const override;
    virtual Variant _get_parameter_default_value(const StringName& p_parameter) const override;
    virtual bool _is_parameter_read_only(const StringName& p_parameter) const override;
    virtual String _get_caption() const override;
    virtual bool _has_filter() const override;

protected:
    void _validate_property(PropertyInfo& p_property) const;
    static void _bind_methods();

private:
    static Dictionary _output_to_dict(const MMQueryOutput& output);
    bool _should_force_query(const MMQueryInput* p_input, double p_delta_time) const;
    struct AnimationInfo {
        StringName name;
        double length = -1;
        double time;
        double delta;
        bool seeked;
        bool is_external_seeking;
        real_t weight;
        real_t blend_spring_speed;
    };

    std::deque<AnimationInfo> _prev_animation_queue;
    AnimationInfo _current_animation_info;

    void _start_transition(const StringName p_animation, float p_time);
    double _update_current_animation(bool p_test_only);

    MMQueryOutput _last_query_output;
    float _time_since_last_query{0.f};

    Vector3 _prev_requested_velocity;
    float _prev_facing;

    double _last_process_time{-1.0};
};

#endif // MM_ANIMATION_NODE_H
