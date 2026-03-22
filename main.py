import src.controller as controller

def print_menu():
    print()
    print("Le funzionalità disponibili sono le seguenti:")
    print("0. Esci dal programma")
    print("1. Genera un nuovo dataset")
    print("2. Visualizza il dataset corrente dopo l'operazione di preprocessing")
    print("3. Addestra i modelli di ML")
    print("4. Visualizza risultati dei modelli di ML sul dataset fornito")
    print("5. Apri la dashboard interattiva")
    print()

def main():

    print("Benvenuto nel programma di machine learning PW20.")
    scelta = ""
    while True:
        print_menu()
        scelta = input("Scegli la funzionalità da eseguire: ")
        match scelta:
            case "0": return
            case "1": controller.generate_dataset()
            case "2": controller.view_preprocessed_dataset()
            case "3": controller.train()
            case "4": controller.check_results()
        print()
        print("====================================================")


if __name__ == "__main__":
    main()
