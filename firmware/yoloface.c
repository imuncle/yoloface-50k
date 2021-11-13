#include "yoloface.h"

#include <stdio.h>

#include "network.h"
#include "network_data.h"

/* Global handle to reference an instantiated C-model */
static ai_handle network = AI_HANDLE_NULL;

/* Global c-array to handle the activations buffer */
AI_ALIGNED(32)
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];


/* 
 * Bootstrap code
 */
int aiInit(void) {
  ai_error err;
  
  /* 1 - Create an instance of the model */
  err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG /* or NULL */);
  if (err.type != AI_ERROR_NONE) {
    printf("E: AI ai_network_create error - type=%d code=%d\r\n", err.type, err.code);
    return -1;
    };

  /* 2 - Initialize the instance */
  const ai_network_params params = AI_NETWORK_PARAMS_INIT(
      AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
      AI_NETWORK_DATA_ACTIVATIONS(activations)
  );

  if (!ai_network_init(network, &params)) {
      err = ai_network_get_error(network);
      printf("E: AI ai_network_init error - type=%d code=%d\r\n", err.type, err.code);
      return -1;
    }

  return 0;
}

/* 
 * Run inference code
 */
int aiRun(const void *in_data, void *out_data)
{
  ai_i32 n_batch;
  ai_error err;

  /* 1 - Create the AI buffer IO handlers with the default definition */
  ai_buffer ai_input[AI_NETWORK_IN_NUM] = AI_NETWORK_IN ;
  ai_buffer ai_output[AI_NETWORK_OUT_NUM] = AI_NETWORK_OUT ;
  
  /* 2 - Update IO handlers with the data payload */
  ai_input[0].n_batches = 1;
  ai_input[0].data = AI_HANDLE_PTR(in_data);
  ai_output[0].n_batches = 1;
  ai_output[0].data = AI_HANDLE_PTR(out_data);

  /* 3 - Perform the inference */
  n_batch = ai_network_run(network, &ai_input[0], &ai_output[0]);
  if (n_batch != 1) {
      err = ai_network_get_error(network);
      printf("E: AI ai_network_run error - type=%d code=%d\r\n", err.type, err.code);
      return -1;
  };
  
  return 0;
}

