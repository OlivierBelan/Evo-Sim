rm *.so
rm -r config_cache
find ../src/ \( -name "*.so" -o -name "*.o" \) -exec gio trash {} +