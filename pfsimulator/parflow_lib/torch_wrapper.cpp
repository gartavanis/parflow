#include "torch_wrapper.h"
#ifdef PARFLOW_HAVE_TORCH

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>

using namespace torch::indexing;

static torch::jit::script::Module model;
static torch::Tensor statics;
static torch::Device device = torch::kCPU;
static torch::Dtype dtype = torch::kDouble;

extern "C" {
  void init_torch_model(char* model_filepath, int nx, int ny, int nz, double *po_dat,
                        double *mann_dat, double *slopex_dat, double *slopey_dat, double *permx_dat,
                        double *permy_dat, double *permz_dat, double *sres_dat, double *ssat_dat,
                        double *fbz_dat, double *specific_storage_dat, double *alpha_dat, double *n_dat,
                        int torch_debug, char* torch_device, char* torch_model_dtype, int torch_include_ghost_nodes) {

    if (std::string(torch_device) == "cuda") {
      if (!torch::cuda::is_available()) {
        throw std::runtime_error("No CUDA device available for Torch!\n");
      }
      device = torch::kCUDA;
    } else if (std::string(torch_device) == "cpu") {
      device = torch::kCPU;
    } else {
      throw std::runtime_error("Invalid Torch device: expected 'cpu' or 'cuda'");
    }

    if (std::string(torch_model_dtype) == "kFloat") {
      dtype = torch::kFloat;
    } else if (std::string(torch_model_dtype) == "kDouble") {
      dtype = torch::kDouble;
    } else {
      throw std::runtime_error("Invalid Torch model dtype: expected 'kFloat' or 'kDouble'");
    }

    c10::InferenceMode guard(true);
    std::string model_path = std::string(model_filepath);
    try {
      model = torch::jit::load(model_path);
      model.to(dtype);
      model.to(device);
      model.eval();
    }
    catch (const c10::Error& e) {
      throw std::runtime_error(std::string("Failed to load the Torch model:\n") + e.what());
    }

    std::unordered_map<std::string, torch::Tensor> statics_map;
    
    // Define slicing based on torch_include_ghost_nodes
    auto z_interior = Slice(1, -1);  // Always exclude ghost nodes in z direction
    auto xy_slice = torch_include_ghost_nodes ? Slice() : Slice(1, -1);  // Conditionally include ghost nodes in x,y

    statics_map["porosity"] = torch::from_blob(po_dat, {nz, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["mannings"] = torch::from_blob(mann_dat, {3, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["slope_x"] = torch::from_blob(slopex_dat, {3, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["slope_y"] = torch::from_blob(slopey_dat, {3, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["perm_x"] = torch::from_blob(permx_dat, {nz, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["perm_y"] = torch::from_blob(permy_dat, {nz, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["perm_z"] = torch::from_blob(permz_dat, {nz, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["sres"] = torch::from_blob(sres_dat, {nz, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["ssat"] = torch::from_blob(ssat_dat, {nz, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["pf_flowbarrier"] = torch::from_blob(fbz_dat, {nz, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["specific_storage"] = torch::from_blob(specific_storage_dat, {nz, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["alpha"] = torch::from_blob(alpha_dat, {nz, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    statics_map["n"] = torch::from_blob(n_dat, {nz, ny, nx}, torch::kDouble)
                                .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);

    statics = model.run_method("get_parflow_statics", statics_map).toTensor();
    if (torch_debug) {
      torch::save(statics, "scaled_statics.pt");
    }
  }

  double* predict_next_pressure_step(double* pp, double* et, double* velx, double* vely, double* velz, int nx, int ny, int nz, int file_number, int torch_debug, int torch_include_ghost_nodes) {
    c10::InferenceMode guard(true);
    
    // Define slicing based on torch_include_ghost_nodes
    auto z_interior = Slice(1, -1);  // Always exclude ghost nodes in z direction
    auto xy_slice = torch_include_ghost_nodes ? Slice() : Slice(1, -1);  // Conditionally include ghost nodes in x,y
    
    torch::Tensor press = torch::from_blob(pp, {nz, ny, nx}, torch::kDouble)
                            .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    torch::Tensor evaptrans = torch::from_blob(et, {nz, ny, nx}, torch::kDouble)
                            .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);

    // Process velocities: velx has shape (z, y, x+1), vely has shape (z, y+1, x), velz has shape (z+1, y, x)
    // Take diff along the velocity dimension to get shape (z, y, x) for each
    // velx: slice z and y, but keep all x values (including the extra one) before taking diff
    torch::Tensor velx_tensor = torch::from_blob(velx, {nz, ny, nx + 1}, torch::kDouble)
                                   .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    // Print the shape of velx_tensor
    std::cout << ">>>>>>>>>> velx_tensor shape: " << velx_tensor.sizes()[0] << " " << velx_tensor.sizes()[1] << " " << velx_tensor.sizes()[2] << std::endl;

    // vely: slice z and x, but keep all y values before taking diff
    torch::Tensor vely_tensor = torch::from_blob(vely, {nz, ny + 1, nx}, torch::kDouble)
                                   .index({z_interior, xy_slice, xy_slice}).clone().to(dtype).to(device);
    // Print the shape of vely_tensor
    std::cout << ">>>>>>>>>> vely_tensor shape: " << vely_tensor.sizes()[0] << " " << vely_tensor.sizes()[1] << " " << vely_tensor.sizes()[2] << std::endl;
    // velz: has 2 ghost nodes in all directions, and one extra layer in z
    // From the actual dimensions: vz_sub has (14, 14, 13) when pressure has (12, 12, 10)
    // Relationship: velz = (nx+2, ny+2, nz+3) = (12+2, 12+2, 10+3) = (14, 14, 13) ✓
    // We need to slice off 2 ghost nodes from all dimensions
    // The blob shape should match the actual velz subvector: (nz+3, ny+2, nx+2) in (z, y, x) order
    auto velz_z_slice = Slice(2, -2);  // Remove 2 ghost nodes from z dimension (both sides)
    auto velz_xy_slice = torch_include_ghost_nodes ? Slice(1, -1) : Slice(2, -2);  // Remove 2 ghost nodes from x,y dimensions (both sides)
    torch::Tensor velz_tensor = torch::from_blob(velz, {nz + 3, ny + 2, nx + 2}, torch::kDouble)
                                   .index({velz_z_slice, velz_xy_slice, velz_xy_slice}).clone().to(dtype).to(device);
    // Print the shape of velz_tensor
    std::cout << ">>>>>>>>>> velz_tensor shape: " << velz_tensor.sizes()[0] << " " << velz_tensor.sizes()[1] << " " << velz_tensor.sizes()[2] << std::endl;
    // Take diff along the velocity dimension
    // velx: diff along x dimension (last dimension) - reduces (z, y, x+1) to (z, y, x)
    velx_tensor = velx_tensor.diff(1, -1);  // diff along last dimension (x)
    std::cout << ">>>>>>>>>> velx_tensor shape after diff: " << velx_tensor.sizes()[0] << " " << velx_tensor.sizes()[1] << " " << velx_tensor.sizes()[2] << std::endl;
    // vely: diff along y dimension (middle dimension) - reduces (z, y+1, x) to (z, y, x)
    vely_tensor = vely_tensor.diff(1, -2);  // diff along second-to-last dimension (y)
    std::cout << ">>>>>>>>>> vely_tensor shape after diff: " << vely_tensor.sizes()[0] << " " << vely_tensor.sizes()[1] << " " << vely_tensor.sizes()[2] << std::endl;
    // velz: diff along z dimension (first dimension) - reduces (z+1, y, x) to (z, y, x)
    velz_tensor = velz_tensor.diff(1, 0);   // diff along first dimension (z)
    std::cout << ">>>>>>>>>> velz_tensor shape after diff: " << velz_tensor.sizes()[0] << " " << velz_tensor.sizes()[1] << " " << velz_tensor.sizes()[2] << std::endl;
    // Concatenate velx, vely, velz along the z dimension (first dimension)
    torch::Tensor velocities = torch::cat({velx_tensor, vely_tensor, velz_tensor}, 0);  // Shape: (3*z, y, x)
    std::cout << ">>>>>>>>>> velocities shape: " << velocities.sizes()[0] << " " << velocities.sizes()[1] << " " << velocities.sizes()[2] << std::endl;
    
    press = model.run_method("get_parflow_pressure", press).toTensor();
    evaptrans = model.run_method("get_parflow_evaptrans", evaptrans).toTensor();
    velocities = velocities.unsqueeze(0);
    velocities = model.run_method("scale_velocity", velocities).toTensor();

    if (torch_debug) {
      char filename[64];
      std::snprintf(filename, sizeof(filename), "scaled_pressure_%05d.pt", file_number);
      torch::save(press, filename);
      std::snprintf(filename, sizeof(filename), "scaled_evaptrans_%05d.pt", file_number);
      torch::save(evaptrans, filename);
    }

    std::vector<torch::jit::IValue> inputs = {press, evaptrans, velocities, statics};
    torch::Tensor output = model.forward(inputs).toTensor();
    torch::Tensor model_output = model.run_method("get_predicted_pressure", output).toTensor()
                                   .to(torch::kCPU).to(torch::kDouble);

    torch::Tensor predicted_pressure = torch::from_blob(pp, {nz, ny, nx}, torch::kDouble);
    predicted_pressure.index_put_({z_interior, xy_slice, xy_slice}, model_output);

    if (!predicted_pressure.is_contiguous()) {
      predicted_pressure = predicted_pressure.contiguous();
    }

    double* predicted_pressure_array = predicted_pressure.data_ptr<double>();
    if (predicted_pressure_array != pp) {
      std::size_t sz = nx * ny * nz;
      std::copy(predicted_pressure_array, predicted_pressure_array + sz, pp);
    }

    return pp;
  }
}

#endif
