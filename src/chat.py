from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta: (digite 'sair' para encerrar)")

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o chat. Até logo!")
            break

        if question.lower() in ("sair", "exit", "quit"):
            print("Encerrando o chat. Até logo!")
            break

        if not question:
            continue

        try:
            resposta = chain(question)
            print(f"RESPOSTA: {resposta}")
        except Exception as exc:  # noqa: BLE001
            print(f"Erro ao processar a pergunta: {exc}")


if __name__ == "__main__":
    main()
