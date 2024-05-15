rm *.so
rm -r config_cache
rm -r __pycache__
find ../src/ \( -name "*.so" -o -name "*.o" -o -name "__pycache__" \) -exec gio trash {} +