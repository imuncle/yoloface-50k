/**
  ******************************************************************************
  * @file    network_config.h
  * @author  AST Embedded Analytics Research Platform
  * @date    Sat Nov  6 11:17:21 2021
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */

#ifndef AI_NETWORK_CONFIG_H
#define AI_NETWORK_CONFIG_H
#pragma once

#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#define AI_TOOLS_VERSION_MAJOR 7
#define AI_TOOLS_VERSION_MINOR 0
#define AI_TOOLS_VERSION_MICRO 0
#define AI_TOOLS_VERSION_EXTRA "RC8"


#undef AI_PLATFORM_API_MAJOR
#undef AI_PLATFORM_API_MINOR
#undef AI_PLATFORM_API_MICRO
#define AI_PLATFORM_API_MAJOR       (1)
#define AI_PLATFORM_API_MINOR       (1)
#define AI_PLATFORM_API_MICRO       (0)

#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#define AI_TOOLS_API_VERSION_MAJOR (1)
#define AI_TOOLS_API_VERSION_MINOR (4)
#define AI_TOOLS_API_VERSION_MICRO (0)

#endif /*AI_NETWORK_CONFIG_H*/
