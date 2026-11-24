import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def ocen_model(nazwa, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{nazwa}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    return accuracy


def testuj_drzewa(x_train, x_test, y_train, y_test):
    wyniki = []

    for kryterium in ["gini", "entropy"]:
        for max_depth in [2, 3, 4, 5, 6, 8, 10, None]:
            model = DecisionTreeClassifier(
                criterion=kryterium,
                max_depth=max_depth,
                random_state=42
            )

            nazwa = f"DecisionTreeClassifier | criterion={kryterium}, max_depth={max_depth}"
            accuracy = ocen_model(nazwa, model, x_train, x_test, y_train, y_test)
            wyniki.append((nazwa, accuracy))

    return wyniki


def testuj_lasy(x_train, x_test, y_train, y_test):
    wyniki = []

    for n_estimators in [10, 30, 50, 100, 200]:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        nazwa = f"RandomForestClassifier | n_estimators={n_estimators}"
        accuracy = ocen_model(nazwa, model, x_train, x_test, y_train, y_test)
        wyniki.append((nazwa, accuracy))

    return wyniki


def pokaz_najlepszy_opis(wyniki, naglowek):
    najlepszy_model = max(wyniki, key=lambda element: element[1])
    print(f"\n{naglowek}")
    print(f"Najlepszy wynik: {najlepszy_model[0]}")
    print(f"Accuracy: {najlepszy_model[1]:.4f}")


def rysuj_wykresy(wyniki_drzew, wyniki_lasow, wynik_logistic, wynik_svm, wynik_voting):
    # Rozdzielam wyniki drzew na gini i entropy, zeby bylo je latwo porownac.
    glebokosci = []
    wyniki_gini = []
    wyniki_entropy = []

    for nazwa, accuracy in wyniki_drzew:
        max_depth = nazwa.split("max_depth=")[-1]
        depth_label = "None" if max_depth == "None" else max_depth

        if "criterion=gini" in nazwa:
            glebokosci.append(depth_label)
            wyniki_gini.append(accuracy)
        else:
            wyniki_entropy.append(accuracy)

    liczba_drzew = []
    wyniki_rf = []

    for nazwa, accuracy in wyniki_lasow:
        n_estimators = int(nazwa.split("=")[-1])
        liczba_drzew.append(n_estimators)
        wyniki_rf.append(accuracy)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(glebokosci, wyniki_gini, marker="o", label="gini")
    axes[0].plot(glebokosci, wyniki_entropy, marker="o", label="entropy")
    axes[0].set_title("Drzewa decyzyjne")
    axes[0].set_xlabel("Max depth")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(liczba_drzew, wyniki_rf, marker="o", color="green")
    axes[1].set_title("Lasy losowe")
    axes[1].set_xlabel("Liczba drzew")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    modele = ["LogisticRegression", "SVM", "Voting"]
    wyniki_koncowe = [wynik_logistic, wynik_svm, wynik_voting]
    kolory = ["orange", "red", "blue"]

    axes[2].bar(modele, wyniki_koncowe, color=kolory)
    axes[2].set_title("Porownanie modeli")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.80, 0.90)
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("wyniki_klasyfikacji.png", dpi=300)
    plt.show()


def main():
    # Tworze dane zgodnie z poleceniem.
    x, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

    # Dziele dane na zbior treningowy i testowy.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print("Rozmiar zbioru treningowego:", len(x_train))
    print("Rozmiar zbioru testowego:", len(x_test))

    print("\n" + "=" * 70)
    print("TESTY DLA DRZEW DECYZYJNYCH")
    print("=" * 70)
    wyniki_drzew = testuj_drzewa(x_train, x_test, y_train, y_test)

    print("\n" + "=" * 70)
    print("TESTY DLA LASOW LOSOWYCH")
    print("=" * 70)
    wyniki_lasow = testuj_lasy(x_train, x_test, y_train, y_test)

    print("\n" + "=" * 70)
    print("REGRESJA LOGISTYCZNA")
    print("=" * 70)
    logistic_model = LogisticRegression(random_state=42, max_iter=1000)
    wynik_logistic = ocen_model(
        "LogisticRegression",
        logistic_model,
        x_train,
        x_test,
        y_train,
        y_test
    )

    print("\n" + "=" * 70)
    print("SVM")
    print("=" * 70)
    svm_model = SVC(kernel="rbf", probability=True, random_state=42)
    wynik_svm = ocen_model(
        "SVC",
        svm_model,
        x_train,
        x_test,
        y_train,
        y_test
    )

    najlepszy_las = max(wyniki_lasow, key=lambda element: element[1])
    najlepsze_n_estimators = int(najlepszy_las[0].split("=")[-1])

    print("\n" + "=" * 70)
    print("VOTING CLASSIFIER")
    print("=" * 70)
    voting_model = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(random_state=42, max_iter=1000)),
            ("rf", RandomForestClassifier(n_estimators=najlepsze_n_estimators, random_state=42)),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
        ],
        voting="soft"
    )
    wynik_voting = ocen_model(
        "VotingClassifier",
        voting_model,
        x_train,
        x_test,
        y_train,
        y_test
    )

    print("\n" + "=" * 70)
    print("PODSUMOWANIE")
    print("=" * 70)
    pokaz_najlepszy_opis(wyniki_drzew, "Najlepsze drzewo decyzyjne:")
    pokaz_najlepszy_opis(wyniki_lasow, "Najlepszy las losowy:")
    print(f"\nLogisticRegression accuracy: {wynik_logistic:.4f}")
    print(f"SVC accuracy: {wynik_svm:.4f}")
    print(f"VotingClassifier accuracy: {wynik_voting:.4f}")

    rysuj_wykresy(
        wyniki_drzew,
        wyniki_lasow,
        wynik_logistic,
        wynik_svm,
        wynik_voting
    )


if __name__ == "__main__":
    main()
