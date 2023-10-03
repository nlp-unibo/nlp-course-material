from pathlib import Path

from cinnamon_generic.api.commands import setup_registry, routine_train

if __name__ == '__main__':
    setup_registry(directory=Path(__file__).parent.parent.resolve(),
                   registrations_to_file=False)

    result = routine_train(name='routine',
                           tags={'task3'},
                           namespace='sp',
                           serialize=True,
                           run_name='test')
