add_executable(encoder_decoder_overfit_test
  encoder_decoder_overfit_test.cpp
)

target_include_directories(encoder_decoder_overfit_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_compile_options(encoder_decoder_overfit_test PRIVATE -fsycl)
target_link_options(encoder_decoder_overfit_test PRIVATE -fsycl)

add_test(
  NAME encoder_decoder_overfit_test
  COMMAND encoder_decoder_overfit_test
)
