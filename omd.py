def step2_umbrella():
    print(
        "утку сдуло ветром на необитаемый остров 🏝️ , где ей налили Кряквенный мартини 🍸"
    )


def step2_no_umbrella():
    print(
        "утка пришла в бар, а по совету бармена устроилась на работу в цирк 🎪 ... маляром 🥁!"
    )


def step1():
    print("Утка-маляр 🦆 решила выпить зайти в бар. Взять ей зонтик? ☂️")
    option = ""
    options = {"да": True, "нет": False}
    while option not in options:
        print("Выберите: {}/{}".format(*options))
        option = input()

    if options[option]:
        return step2_umbrella()
    return step2_no_umbrella()


if __name__ == "__main__":
    step1()
