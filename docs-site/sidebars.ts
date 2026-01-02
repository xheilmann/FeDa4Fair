import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Sidebar configuration for FeDa4Fair documentation.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'API',
      items: [
        'api/FairFederatedDataset',
        'api/Partitioning',
        'api/FairnessComputation',
        'api/Evaluation',
        'api/Plots',
        'api/Utils',
        'api/CreatingDatasets',
        'api/CustomDatasets',
        'api/Datasheets',
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      items: [
        'examples/ACSIncome',
        'examples/CelebA',
        'examples/DutchCensus',
        'examples/DutchAttribute',
        'examples/DutchValue',
        'examples/PreLoadedData',
      ],
    },
    {
      type: 'category',
      label: 'Benchmarking Datasets',
      items: [
        'benchmarking-datasets/Datasets',
      ],
    },
  ],
};

export default sidebars;