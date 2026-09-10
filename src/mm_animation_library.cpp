#include "mm_animation_library.h"

#include "common.h"
#include "features/mm_feature.h"
#include "math/hash.h"
#include "math/stats.hpp"
#include "mm_character.h"

#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <limits>
void MMAnimationLibrary::bake_data(const MMCharacter* p_character, const AnimationMixer* p_player, const Skeleton3D* p_skeleton) {
    ERR_FAIL_COND(features.is_empty());

    motion_data.clear();
    db_anim_index.clear();
    db_time_index.clear();
    db_pose_offset.clear();

    int32_t dim_count = 0;
    for (auto i = 0; i < features.size(); ++i) {
        MMFeature* f = Object::cast_to<MMFeature>(features[i]);
        dim_count += f->get_dimension_count();
        f->setup_skeleton(p_character, p_player, p_skeleton);
    }

    TypedArray<StringName> animation_list = get_animation_list();

    // Normalization data
    std::vector<std::vector<StatsAccumulator>> stats(features.size());

    PackedFloat32Array data;
    int32_t current_pose_offset = 0;
    // For every animation
    for (int64_t animation_index = 0; animation_index < animation_list.size(); animation_index++) {
        const StringName& anim_name = animation_list[animation_index];
        Ref<Animation> animation = get_animation(anim_name);
        db_pose_offset.push_back(current_pose_offset);

        // Initialize features
        for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
            MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
            stats[feature_index].resize(feature->get_dimension_count());
            feature->setup_for_animation(animation);
        }

        const double animation_length = animation->get_length();
        const double time_step = 1.0f / get_sampling_rate();
        int pose_count = 0;
        // Every time step
        for (double time = 0; time < animation_length; time += time_step) {
            PackedFloat32Array pose_data;
            // For every feature
            for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
                const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
                const PackedFloat32Array feature_data = feature->bake_animation_pose(animation, time);

                if (feature_data.size() != feature->get_dimension_count()) {
                    print_line("[MM] bake mismatch feature_index=", feature_index,
                        " data=", feature_data.size(),
                        " dim=", feature->get_dimension_count(),
                        " anim=", String(anim_name),
                        " time=", String::num(time, 3));
                }

                ERR_FAIL_COND(feature_data.size() != feature->get_dimension_count());

                // Update stats
                for (int64_t feature_element_index = 0; feature_element_index < feature_data.size(); feature_element_index++) {
                    stats[feature_index][feature_element_index].add_sample(feature_data[feature_element_index]);
                }

                pose_data.append_array(feature_data);
                current_pose_offset += feature->get_dimension_count();
            }

            ERR_FAIL_COND_MSG(pose_data.size() != dim_count,
                "[MM] bake aggregate mismatch pose=" + String::num(pose_data.size()) +
                " dim_count=" + String::num(dim_count) +
                " anim=" + String(anim_name));
            if (pose_data.size() != dim_count) {
                print_line("[MM] bake aggregate mismatch pose=", pose_data.size(),
                    " dim_count=", dim_count, " anim=", String(anim_name),
                    " time=", String::num(time, 3), " features=", features.size());
            }

            // Update dataset
            data.append_array(pose_data);
            db_anim_index.push_back(animation_index);
            db_time_index.push_back(time);
            pose_count++;
        }
    }

    // Compute mean and standard deviation
    for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
        MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);

        PackedFloat32Array feature_means;
        feature_means.resize(feature->get_dimension_count());
        PackedFloat32Array feature_std_devs;
        feature_std_devs.resize(feature->get_dimension_count());
        PackedFloat32Array feature_mins;
        feature_mins.resize(feature->get_dimension_count());
        PackedFloat32Array feature_maxes;
        feature_maxes.resize(feature->get_dimension_count());

        for (int64_t feature_element_index = 0; feature_element_index < feature->get_dimension_count(); feature_element_index++) {
            feature_means.set(feature_element_index, stats[feature_index][feature_element_index].get_mean());
            feature_std_devs.set(feature_element_index, stats[feature_index][feature_element_index].get_standard_deviation());
            feature_mins.set(feature_element_index, stats[feature_index][feature_element_index].get_min());
            feature_maxes.set(feature_element_index, stats[feature_index][feature_element_index].get_max());
        }

        feature->set_means(feature_means);
        feature->set_std_devs(feature_std_devs);
        feature->set_mins(feature_mins);
        feature->set_maxes(feature_maxes);
    }

    // Guard against an inconsistent bake (empty / non-divisible data). Building a KDTree
    // on empty data dereferences a null pointer, which terminates the process — abort
    // cleanly here instead of crashing the engine.
    if (data.is_empty() || dim_count <= 0 || (data.size() % dim_count != 0)) {
        ERR_FAIL_COND_MSG(true, "MMAnimationLibrary::bake_data produced invalid data (size="
            + String::num_int64(data.size()) + ", dim_count=" + String::num_int64(dim_count)
            + "); check that each feature's dimension count matches what it bakes.");
        return;
    }

    _normalize_data(data, dim_count);

    motion_data = data.duplicate();

    schema_hash = compute_features_hash();

    _kd_tree = std::make_unique<KDTree>(motion_data.ptr(), dim_count, ((int32_t)motion_data.size()) / dim_count);
    node_indices = PackedInt32Array(_kd_tree->get_node_indices());
}

MMQueryOutput MMAnimationLibrary::query(const MMQueryInput& p_query_input) {
    PackedFloat32Array query_vector = PackedFloat32Array();
    for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
        const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
        if (!feature) {
            continue;
        }
        PackedFloat32Array feature_data = feature->evaluate_runtime_data(p_query_input);
        feature->normalize(feature_data.ptrw());
        query_vector.append_array(feature_data);
    }

    MMQueryOutput result;
    if (lmm_enabled) {
        result = _search_lmm(query_vector);
    } else {
        result = _search_kd_tree(query_vector);
    }
    return std::move(result);
}

int64_t MMAnimationLibrary::get_dim_count() const {
    int64_t dim_count = 0;
    for (auto i = 0; i < features.size(); ++i) {
        MMFeature* f = Object::cast_to<MMFeature>(features[i]);
        dim_count += f->get_dimension_count();
    }

    return dim_count;
}

int64_t MMAnimationLibrary::get_animation_pose_count(String p_animation_name) const {
    TypedArray<StringName> animation_list = get_animation_list();
    Ref<Animation> animation = get_animation(p_animation_name);
    if (animation.is_null()) {
        return 0;
    };
    const double animation_length = animation->get_length();
    const double time_step = 1.0f / get_sampling_rate();
    return static_cast<int32_t>(UtilityFunctions::floor(animation_length / time_step));
}

void MMAnimationLibrary::display_data(const Ref<EditorNode3DGizmo>& p_gizmo, const Transform3D& p_transform, String p_animation_name, int32_t p_pose_index) const {
    const int32_t anim_index = get_animation_list().find(p_animation_name);
    const int32_t dim_count = get_dim_count();
    int32_t start_frame_index = db_pose_offset[anim_index];
    int32_t frame_index = start_frame_index + p_pose_index * dim_count;

    for (size_t feature_index = 0; feature_index < features.size(); feature_index++) {
        const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
        const float* frame_motion_data = motion_data.ptr() + frame_index;
        feature->display_data(p_gizmo, p_transform, frame_motion_data);
        frame_index += feature->get_dimension_count();
    }
}

int64_t MMAnimationLibrary::compute_features_hash() const {
    int64_t hash = 0;
    for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
        MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
        TypedArray<Dictionary> feature_properties = feature->get_property_list();
        for (int64_t property_index = 0; property_index < feature_properties.size(); property_index++) {
            // TODO: This needs work, but works for now
            const Dictionary feature_property = Dictionary(feature_properties[property_index]);
            const String property_name = feature_property["name"];
            const uint32_t property_type = feature_property["type"];
            const bool property_is_stats =
                property_name == "means" ||
                property_name == "std_devs" ||
                property_name == "mins" ||
                property_name == "maxes";
            const bool is_resource_property = property_name.contains("resource");
            if (property_type != Variant::OBJECT &&
                property_type != Variant::NIL &&
                !property_is_stats &&
                !is_resource_property) {
                hash = hash_combine(hash, property_name.hash());
                hash = hash_combine(hash, feature->get(property_name).hash());
            }
        }
    }
    return hash;
}

bool MMAnimationLibrary::needs_baking() const {
    return schema_hash != compute_features_hash();
}

void MMAnimationLibrary::_normalize_data(PackedFloat32Array& p_data, size_t p_dim_count) const {
    ERR_FAIL_COND(p_data.is_empty() || (p_data.size() % p_dim_count != 0));

    for (int64_t frame_index = 0; frame_index < p_data.size(); frame_index += p_dim_count) {

        int dim_index = 0;
        for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
            const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
            feature->normalize(p_data.ptrw() + frame_index + dim_index);
            dim_index += feature->get_dimension_count();
        }
    }
}

float MMAnimationLibrary::_compute_feature_costs(int p_pose_index, const PackedFloat32Array& p_query, Dictionary* p_feature_costs) const {
    float pose_cost = 0.f;
    int start_frame_index = p_pose_index * p_query.size();
    int start_feature_index = 0;
    for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
        const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
        if (!feature) {
            continue;
        }

        float feature_cost = 0.f;
        for (int64_t feature_dim_index = 0; feature_dim_index < feature->get_dimension_count(); feature_dim_index++) {
            feature_cost += distance_squared((motion_data.ptr() + start_frame_index + start_feature_index + feature_dim_index),
                                             (p_query.ptr() + start_feature_index + feature_dim_index),
                                             1) *
                feature->calculate_normalized_weight(feature_dim_index);
        }

        if (p_feature_costs) {
            p_feature_costs->get_or_add(feature->get_class(), feature_cost);
        }

        pose_cost += feature_cost;
        start_feature_index += feature->get_dimension_count();
    }
    return pose_cost;
}

MMQueryOutput MMAnimationLibrary::_search_naive(const PackedFloat32Array& p_query) const {
    float cost = FLT_MAX;
    MMQueryOutput result;
    TypedArray<StringName> animation_list = get_animation_list();
    int32_t dim_count = p_query.size();
    int64_t best_pose_index = -1;
    for (int64_t start_frame_index = 0; start_frame_index < motion_data.size(); start_frame_index += dim_count) {
        int start_feature_index = start_frame_index;
        float pose_cost = 0.f;
        Dictionary feature_costs;
        for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
            const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
            if (!feature) {
                continue;
            }

            float feature_cost = 0.f;
            for (int64_t feature_dim_index = 0; feature_dim_index < feature->get_dimension_count(); feature_dim_index++) {
                feature_cost += distance_squared((motion_data.ptr() + start_frame_index + start_feature_index + feature_dim_index),
                                                 (p_query.ptr() + start_feature_index + feature_dim_index),
                                                 1) *
                    feature->calculate_normalized_weight(feature_dim_index);
            }

            feature_costs.get_or_add(feature->get_class(), feature_cost);
            pose_cost += feature_cost;
            start_feature_index += feature->get_dimension_count();
        }

        if (pose_cost < cost) {
            cost = pose_cost;
            best_pose_index = start_frame_index / dim_count;
        }
    }

    String library_name = get_path().get_file().get_basename() + "/";
    if (library_name.is_empty()) {
        library_name = get_name() + "/";
    }

    result.matched_pose_index = best_pose_index;
    result.animation_match = library_name + UtilityFunctions::str(animation_list[db_anim_index[best_pose_index]]);
    result.time_match = db_time_index[best_pose_index];

    if (include_cost_results) {
        result.matched_frame_data = motion_data.slice(
            best_pose_index * dim_count,
            (best_pose_index + 1) * dim_count);

        result.cost = _compute_feature_costs(
            best_pose_index,
            p_query,
            &result.feature_costs);
    }

    return result;
}

MMQueryOutput MMAnimationLibrary::_search_kd_tree(const PackedFloat32Array& p_query) {
    if (!_kd_tree) {
        int32_t dim_count = p_query.size();
        _kd_tree = std::make_unique<KDTree>(dim_count);
        _kd_tree->rebuild_tree(((int32_t)motion_data.size()) / dim_count, node_indices);
    }

    std::vector<float> dimension_weights;
    for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
        const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
        if (!feature) {
            continue;
        }
        for (int64_t feature_dim_index = 0; feature_dim_index < feature->get_dimension_count(); feature_dim_index++) {
            dimension_weights.push_back(feature->calculate_normalized_weight(feature_dim_index));
        }
    }

    // Guard against a feature/query dimension mismatch: _search_nn_recursive reads
    // dimension_weigths[i] for i in [0, dim), so a shorter vector OOBs and the abort
    // pops out of Godot's error handler (seen as 'MMAnimationLibrary::_search_kd_tree'
    // abort). Bail to a safe index instead of searching with a broken weight vector.
    const int32_t dim_count = (int32_t)p_query.size();
    if (dim_count != (int32_t)dimension_weights.size() || dim_count <= 0) {
        MMQueryOutput result;
        result.matched_pose_index = -1;
        return result;
    }

    int nodes_visited = 0;
    int best_pose_index = _kd_tree->search_nn(
        motion_data.ptr(),
        p_query.ptr(),
        dimension_weights);

    // search_nn can return -1 (empty/invalid tree); guard before indexing the DB so a
    // bad index aborts Godot instead of returning a poison pose.
    if (best_pose_index < 0 || best_pose_index >= db_anim_index.size()) {
        MMQueryOutput result;
        result.matched_pose_index = best_pose_index;
        return result;
    }

    MMQueryOutput result;
    String library_name = get_path().get_file().get_basename() + "/";
    if (library_name.is_empty()) {
        library_name = get_name() + "/";
    }
    TypedArray<StringName> animation_list = get_animation_list();

    result.matched_pose_index = best_pose_index;
    result.animation_match = library_name + UtilityFunctions::str(animation_list[db_anim_index[best_pose_index]]);
    result.time_match = db_time_index[best_pose_index];

    if (include_cost_results) {
        result.matched_frame_data = motion_data.slice(
            best_pose_index * p_query.size(),
            (best_pose_index + 1) * p_query.size());

        result.cost =
            _compute_feature_costs(
                best_pose_index,
                p_query,
                &result.feature_costs);
    }

    return result;
}

MMQueryOutput MMAnimationLibrary::_search_lmm(const PackedFloat32Array& p_query) {
    MMQueryOutput empty;
    const int dim = (int)p_query.size();
    if (dim <= 0 || motion_data.size() == 0 || (motion_data.size() % dim) != 0) {
        empty.matched_pose_index = -1;
        return empty;
    }
    const int row_count = (int)(motion_data.size() / dim);

    // Load the projector + decompressor nets once.
    if (!_lmm_loaded) {
        const String proj = lmm_projector_path;
        const String deco = lmm_decompressor_path;
        const String proj_base = proj.is_empty() ? "res://assets/mm/lmm/projector.bin" : proj;
        const String deco_base = deco.is_empty() ? "res://assets/mm/lmm/decompressor.bin" : deco;
        // fopen needs a filesystem path; globalize res:// -> absolute.
        const String proj_fs = ProjectSettings::get_singleton()->globalize_path(proj_base);
        const String deco_fs = ProjectSettings::get_singleton()->globalize_path(deco_base);
        if (!_projector.load(proj_fs.utf8().get_data()) ||
            !_decompressor.load(deco_fs.utf8().get_data())) {
            ERR_PRINT("[MM] LMM failed to load projector/decompressor; falling back to KD-tree.");
            empty.matched_pose_index = -1;
            return empty;
        }
        _lmm_loaded = true;
    }

    const int dim_in = _projector.in_dim;
    const int latent_dim = _projector.out_dim;
    if (dim != dim_in || _decompressor.in_dim != latent_dim || _decompressor.out_dim != dim) {
        empty.matched_pose_index = -1;
        return empty;
    }

    // query -> latent (projector), latent -> feature (decompressor).
    std::vector<float> latent(latent_dim);
    _projector.evaluate(p_query.ptr(), latent.data());
    std::vector<float> decoded(dim);
    _decompressor.evaluate(latent.data(), decoded.data());

    // nearest motion_data row to the decoded feature -> pose index.
    int best = -1;
    float best_d = std::numeric_limits<float>::max();
    for (int r = 0; r < row_count; r++) {
        const float* row = motion_data.ptr() + (size_t)r * dim;
        float d = 0.0f;
        for (int i = 0; i < dim; i++) {
            float diff = row[i] - decoded[i];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; best = r; }
    }
    if (best < 0 || best >= db_anim_index.size()) {
        empty.matched_pose_index = best;
        return empty;
    }

    MMQueryOutput result;
    String library_name = get_path().get_file().get_basename() + "/";
    if (library_name.is_empty()) {
        library_name = get_name() + "/";
    }
    TypedArray<StringName> animation_list = get_animation_list();
    result.matched_pose_index = best;
    result.animation_match = library_name + UtilityFunctions::str(animation_list[db_anim_index[best]]);
    result.time_match = db_time_index[best];
    if (include_cost_results) {
        result.matched_frame_data = motion_data.slice(best * dim, (best + 1) * dim);
    }
    return result;
}

void MMAnimationLibrary::_bind_methods() {
    ClassDB::bind_method(D_METHOD("bake_data", "character", "player", "skeleton"), &MMAnimationLibrary::bake_data);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::ARRAY, features, PROPERTY_HINT_TYPE_STRING, UtilityFunctions::str(Variant::OBJECT) + '/' + UtilityFunctions::str(Variant::BASIS) + ":MMFeature");
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::FLOAT, sampling_rate);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::BOOL, include_cost_results);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_FLOAT32_ARRAY, motion_data, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_INT32_ARRAY, db_anim_index, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_FLOAT32_ARRAY, db_time_index, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_FLOAT32_ARRAY, db_pose_offset, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::INT, schema_hash, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_INT32_ARRAY, node_indices, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::BOOL, lmm_enabled);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::STRING, lmm_decompressor_path);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::STRING, lmm_projector_path);
}
