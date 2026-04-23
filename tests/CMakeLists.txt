add_executable(encoder_decoder_overfit_test
  encoder_decoder_overfit_test.cpp
  ../src/long_term/spatial_map.cpp
  ../src/long_term/graph.cpp
)

target_include_directories(encoder_decoder_overfit_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_compile_options(encoder_decoder_overfit_test PRIVATE -fsycl -mavx2 -mf16c)
target_link_options(encoder_decoder_overfit_test PRIVATE -fsycl)

add_test(
  NAME encoder_decoder_overfit_test
  COMMAND encoder_decoder_overfit_test
)

add_executable(looping_retnet_serialize_test
  looping_retnet_serialize_test.cpp
)

target_include_directories(looping_retnet_serialize_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_compile_options(looping_retnet_serialize_test PRIVATE -fsycl)
target_link_options(looping_retnet_serialize_test PRIVATE -fsycl)

add_test(
  NAME looping_retnet_serialize_test
  COMMAND looping_retnet_serialize_test
)

add_executable(looping_retnet_training_test
  looping_retnet_training_test.cpp
  ../src/long_term/spatial_map.cpp
  ../src/long_term/graph.cpp
)

target_include_directories(looping_retnet_training_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_compile_options(looping_retnet_training_test PRIVATE -fsycl -mavx2 -mf16c)
target_link_options(looping_retnet_training_test PRIVATE -fsycl)

add_test(
  NAME looping_retnet_training_test
  COMMAND looping_retnet_training_test
)

add_executable(looping_retnet_training_leak_smoke_test
  looping_retnet_training_leak_smoke_test.cpp
  ../src/long_term/spatial_map.cpp
  ../src/long_term/graph.cpp
)

target_include_directories(looping_retnet_training_leak_smoke_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_compile_options(looping_retnet_training_leak_smoke_test PRIVATE -fsycl -mavx2 -mf16c)
target_link_options(looping_retnet_training_leak_smoke_test PRIVATE -fsycl)

add_test(
  NAME looping_retnet_training_leak_smoke_test
  COMMAND looping_retnet_training_leak_smoke_test
)
