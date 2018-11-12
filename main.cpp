#include <vector>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <algorithm>
//#include <fstream>
#include "CL\CL2.hpp"
using namespace std;

#define INCLUDE_SPEED_TEST
#define INCLUDE_UNIVARIATE_TESTS
#define INCLUDE_CORRELATION_TEST

string file_to_string(char* filename) {
	FILE* f = fopen(filename, "rb");
	if (f) {
		fseek(f, 0L, SEEK_END);
		unsigned int length = ftell(f);
		fseek(f, 0L, SEEK_SET);
		char* s = (char*)malloc(sizeof(char)*(length + 1));
		if (s)
			fread(s, 1, length, f);
		else return NULL;
		fclose(f);
		s[length] = '\0';
		return string(s);
	}
	else return NULL;
}

void handle_error(cl_int error_code, char* s) {
	if (CL_SUCCESS != error_code) {
		printf(s, error_code);
	    exit(EXIT_FAILURE);
	}
}

void handle_program_build_errors(cl_int error_code, cl_program program, cl_device_id device) {
	if (CL_SUCCESS != error_code) {
		char* build_log;
		size_t n_bytes;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &n_bytes);
		build_log = (char*)malloc(sizeof(char)*n_bytes);
		if (build_log) {
			cl_int new_error_code = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(char)*n_bytes, build_log, &n_bytes);
			printf("%s", build_log);
			free(build_log);
			handle_error(new_error_code, "clGetProgramBuildInfo(...) failed with error code %d.\n");
		}
		else printf("Out of memory.\n");
	}
}

int main(int argc, char** argv) {
	cl_int error_code;

	//cl::Platform platform = cl::Platform::getDefault(&error_code);
	//handle_error(error_code, "Could not get default platform. cl::Platform::getDefault(...) failed with error code %d.\n");

	string source_code = file_to_string("Randomness.cl");
	cl::Program program(source_code, true, &error_code);
	handle_error(error_code, "Could not build program.\n cl::Program constructor failed with error %d.\n");

	cl::Kernel speed_test_kernel(program, "speed_test", &error_code);
	handle_error(error_code, "Could not construct speed test kernel.\n cl::Kernel constructor failed with error %d.\n");

	cl_int N = 500000;//20000000;

	//error_code = speed_test_kernel.setArg(0, mode);
	//handle_error(error_code, "Could not set first kernel argument.\n cl::Kernel::setArg(...) failed with error code %d.\n");

	error_code = speed_test_kernel.setArg(1, N);
	handle_error(error_code, "Could not set second kernel argument.\n cl::Kernel::setArg(...) failed with error code %d.\n");

	cl::Device device = cl::Device::getDefault(&error_code);
	handle_error(error_code, "Could not get default device.\n cl::Device::getDefault(...) failed with error code %d.\n");

	cl::Context context = cl::Context::getDefault(&error_code);
	handle_error(error_code, "Could not get default context.\n cl::Context::getDefault(...) failed with error code %d.\n");

	cl::CommandQueue command_queue(context, device, cl::QueueProperties::None, &error_code);
	handle_error(error_code, "Could not create command queue. cl::DeviceCommandQueue constructor failed with error code %d.\n");

	cl::Kernel moment_test_kernel(program, "test_moment", &error_code);
	handle_error(error_code, "Could not construct moment test kernel.\n cl::Kernel constructor failed with error %d.\n");

	cl::Kernel correlation_test_kernel(program, "test_correlation", &error_code);
	handle_error(error_code, "Could not construct correlation test kernel.\n cl::Kernel constructor failed with error %d.\n");

	char* s[] = { "Hexagonal Marsaglia polar method", "Marsaglia polar method", "Box-Muller transform" };
#ifdef INCLUDE_SPEED_TEST
	for (cl_int i = 0; i != 3; ++i) {
		speed_test_kernel.setArg(0, i + 1);
		auto start = chrono::high_resolution_clock::now();
		command_queue.enqueueNDRangeKernel(speed_test_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange);
		command_queue.finish();
		auto finish = chrono::high_resolution_clock::now();
		cout << s[i] << " finished in " << (float)(finish - start).count() / 1000000.0f << " ms.\n";
	}
	cout << "\n\n";
#endif

#if defined(INCLUDE_UNIVARIATE_TESTS) || defined(INCLUDE_CORRELATION_TEST)
	size_t workgroup_size1 = moment_test_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device, &error_code);
	handle_error(error_code, "Could not get moment test kernel workgroup size. cl::Kernel::getWorkGroupInfo<CL_KERNEL_GLOBAL_WORK_SIZE>(...) failed with error code %d.\n");

	size_t workgroup_size2 = correlation_test_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device, &error_code);
	handle_error(error_code, "Could not get correlation test kernel workgroup size. cl::Kernel::getWorkGroupInfo<CL_KERNEL_GLOBAL_WORK_SIZE>(...) failed with error code %d.\n");

	size_t output_buffer_size  = min(workgroup_size1, workgroup_size2),
		   oversize_allocation = 64 * (output_buffer_size / 64 + 1);

	cl_float* output = (cl_float*)_aligned_malloc(sizeof(cl_float)*oversize_allocation, 4096);
	if (!output) {
		cout << "Could not allocate memory for output buffer.\n";
		exit(EXIT_FAILURE);
	}
	cl::Buffer output_buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float)*oversize_allocation, (void*)output, &error_code);
	handle_error(error_code, "Could not create buffer. cl::Buffer constructor failed with error code %d.\n");
#endif
#ifdef INCLUDE_UNIVARIATE_TESTS
	moment_test_kernel.setArg(0, output_buffer);
	//moment_test_kernel.setArg(1, 1); //p
	//moment_test_kernel.setArg(2, 1); //mode
	moment_test_kernel.setArg(3, N/output_buffer_size + 1);

	char* t[] = {"mean", "variance", "skewness"};
	for (cl_int i = 0; i != 3; ++i) {
		for (cl_int p = 1; p != 4; ++p) {
			moment_test_kernel.setArg(1, p); //p
			moment_test_kernel.setArg(2, i + 1); //mode

			error_code = command_queue.enqueueNDRangeKernel(moment_test_kernel, cl::NullRange, cl::NDRange(output_buffer_size), cl::NullRange);
			handle_error(error_code, "Could not enqueue moment test kernel.\n CommandQueue::enqueueNDRangeKernel(...) failed with error code %d.\n");

			error_code = command_queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_buffer_size, output);
			handle_error(error_code, "Could not read buffers from device.\n CommandQueue::enqueuReadBuffer(...) failed with error code %d.\n");

			command_queue.finish();

			float sum = 0.0f;
			for (int i = 0; i != output_buffer_size; ++i)
				sum += output[i];
			sum /= (float)output_buffer_size;

			cout << s[i] << " produced " << t[p - 1] << " " << sum << endl;
		}
		cout << endl;
	}
#endif

#ifdef INCLUDE_CORRELATION_TEST
	correlation_test_kernel.setArg(0, output_buffer);
	correlation_test_kernel.setArg(2, N / output_buffer_size + 1); //N
	for (cl_int i = 0; i != 3; ++i) {
		correlation_test_kernel.setArg(1, i + 1); //mode

		error_code = command_queue.enqueueNDRangeKernel(correlation_test_kernel, cl::NullRange, cl::NDRange(output_buffer_size), cl::NullRange);
		handle_error(error_code, "Could not enqueue moment test kernel.\n CommandQueue::enqueueNDRangeKernel(...) failed with error code %d.\n");

		error_code = command_queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_buffer_size, output);
		handle_error(error_code, "Could not read buffers from device.\n CommandQueue::enqueuReadBuffer(...) failed with error code %d.\n");

		float sum = 0.0f;
		for (int i = 0; i != output_buffer_size; ++i)
			sum += output[i];
		sum /= (float)output_buffer_size;

		cout << s[i] << " produced correlation " << sum << endl;
	}
#endif

	_aligned_free(output);

	return EXIT_SUCCESS;
}