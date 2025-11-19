def convert_toml_to_tuple_style(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    _ = lines.pop(0)
    converted = []
    for line in lines:
        if "=" in line and "[" in line and "]" in line:
            key, value = line.split("=", 1)
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                # Replace brackets with parentheses and remove spaces before commas
                tuple_value = "(" + value[1:-1].strip() + ")"
                converted.append(f"{key.strip()} = \"{tuple_value}\"\n")
            else:
                converted.append(line)
        else:
            converted.append(line)

    with open(output_path, 'w') as f:
        f.writelines(converted)

    print(f"✅ Converted TOML saved to {output_path}")

# Example usage:
convert_toml_to_tuple_style("scenario1.toml", "typst/scenario1.toml")
convert_toml_to_tuple_style("scenario2.toml", "typst/scenario2.toml")
convert_toml_to_tuple_style("scenario3.toml", "typst/scenario3.toml")
convert_toml_to_tuple_style("scenario4.toml", "typst/scenario4.toml")
convert_toml_to_tuple_style("scenario5.toml", "typst/scenario5.toml")
convert_toml_to_tuple_style("scenario6.toml", "typst/scenario6.toml")
