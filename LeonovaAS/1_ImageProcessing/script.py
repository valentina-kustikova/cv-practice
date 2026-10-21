from util import cli_argument_parser, parse_args, read_image, save_image
from filter_base import ImageFilter
 
def main():
    args = cli_argument_parser()
    kwargs = parse_args(args.filter_args)
    image = read_image(args.image)
    
    f1 = ImageFilter.get_filter(args.filter, **kwargs)
    result = f1.apply_filter(image)
    
    save_image(args.filter, args.image, result)

if __name__ == "__main__":
    main()   