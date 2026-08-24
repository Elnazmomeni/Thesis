import click

@click.command()
@click.option('--device', required=True, help='Target device')
def main(device):
    print(f"Hello World, the device is {device}!")

if __name__ == '__main__':
    main()