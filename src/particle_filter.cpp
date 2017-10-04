/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <random>
#include <iostream>
#include <tuple>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using std::normal_distribution;
using std::default_random_engine;

using vector_t = std::vector<double>;

default_random_engine gen;

/**
 * Dynamic model
 * @param delta_t
 * @param x State vector [x, y, theta]
 * @param u Control vector [v and yaw-dot]
 * @param std_pos Standard deviation of the noise in state
 * @return Predicted state
 */
const vector_t CTRV_ModelFunc(double delta_t, const vector_t &x, const vector_t &u, const double std_pos[]) {
    static default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_yaw(0, std_pos[2]);

    //extract values for better readability
    double p_x = x[0];
    double p_y = x[1];
    double yaw = x[2];

    double v = u[0];
    double yawd = u[1];

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (std::fabs(yawd) > 0.01) {
        px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }
    // Yaw angle
    double pyaw_p = yaw + yawd * delta_t;

    // Add noise
    px_p = px_p + dist_x(gen);
    py_p = py_p + dist_y(gen);
    pyaw_p = pyaw_p + dist_yaw(gen);

    //return predicted state
    return {px_p, py_p, pyaw_p};
}

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Add random Gaussian noise to each particle.
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    double sample_x, sample_y, sample_psi;

    num_particles = 200;
    weights.resize(num_particles, 1.0f);

    for (unsigned i = 0; i < num_particles; i++) {
        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.id = i;
        p.weight = 1.0f;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    for (auto i = 0; i < particles.size(); i++) {
        auto p = particles[i];
        vector_t x_pred = CTRV_ModelFunc(delta_t, {p.x, p.y, p.theta}, {velocity, yaw_rate}, std_pos);
        particles[i].x = x_pred[0];
        particles[i].y = x_pred[1];
        particles[i].theta = x_pred[2];
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    double min_distance, dist, dx, dy;
    int min_i;

    for (unsigned obs_i = 0; obs_i < observations.size(); obs_i++) {
        auto obs = observations[obs_i];

        min_distance = INFINITY;
        min_i = -1;
        for (unsigned i = 0; i < predicted.size(); i++) {
            auto pred_lm = predicted[i];
            dx = (pred_lm.x - obs.x);
            dy = (pred_lm.y - obs.y);
            dist = dx * dx + dy * dy;
            if (dist < min_distance) {
                min_distance = dist;
                min_i = i;
            }
        }
        observations[obs_i].id = min_i; // Use index of landmark as the ID (rather than the id field)
    }
}

const LandmarkObs local_to_global(const LandmarkObs &obs, const Particle &p) {
    LandmarkObs out;

    // First rotate the local coordinates to the right orientation
    out.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
    out.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
    out.id = obs.id;
    return out;
}

inline const double gaussian_2d(const LandmarkObs &obs, const LandmarkObs &lm, const double sigma[]) {
    auto cov_x = sigma[0] * sigma[0];
    auto cov_y = sigma[1] * sigma[1];
    auto normalizer = 2.0 * M_PI * sigma[0] * sigma[1];
    auto dx = (obs.x - lm.x);
    auto dy = (obs.y - lm.y);
    return exp(-(dx * dx / (2 * cov_x) + dy * dy / (2 * cov_y))) / normalizer;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    double sigma_landmark[2] = {0.3, 0.3}; // Landmark measurement uncertainty [x [m], y [m]]

    for (unsigned p_ctr = 0; p_ctr < particles.size(); p_ctr++) {
        auto p = particles[p_ctr];

        std::vector<LandmarkObs> predicted_landmarks;

        for (auto lm : map_landmarks.landmark_list) {
            LandmarkObs lm_pred;
            lm_pred.x = lm.x_f;
            lm_pred.y = lm.y_f;
            lm_pred.id = lm.id_i;
            auto dx = lm_pred.x - p.x;
            auto dy = lm_pred.y - p.y;

            // Add only if in range
            if (dx * dx + dy * dy <= sensor_range * sensor_range)
                predicted_landmarks.push_back(lm_pred);
        }
        std::vector<LandmarkObs> transformed_obs;
        double total_prob = 1.0f;

        // transform coordinates of all observations (for current particle)
        for (auto obs_lm : observations) {
            auto obs_global = local_to_global(obs_lm, p);
            transformed_obs.push_back(std::move(obs_global));
        }
        // Stores index of associated landmark in the observation
        dataAssociation(predicted_landmarks, transformed_obs);

        for (unsigned i = 0; i < transformed_obs.size(); i++) {
            auto obs = transformed_obs[i];
            // Assume sorted by id and starting at 1
            auto assoc_lm = predicted_landmarks[obs.id];

            double pdf = gaussian_2d(obs, assoc_lm, sigma_landmark);
            total_prob *= pdf;
        }
        particles[p_ctr].weight = total_prob;
        weights[p_ctr] = total_prob;
    }
    std::cout << std::endl;
}

void ParticleFilter::resample() {
    std::discrete_distribution<int> d(weights.begin(), weights.end());
    std::vector<Particle> new_particles;

    for (unsigned i = 0; i < num_particles; i++) {
        auto ind = d(gen);
        new_particles.push_back(std::move(particles[ind]));
    }
    particles = std::move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
