import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Federated Datasets',
    Svg: require('@site/static/img/icon_federated.svg').default,
    description: (
      <>
        Easily create federated datasets for <b>Cross-Silo</b> and <b>Cross-Device</b> settings.
        Simulate realistic client distributions with flexible partitioning strategies.
      </>
    ),
  },
  {
    title: 'Fairness Benchmarking',
    Svg: require('@site/static/img/icon_fairness.svg').default,
    description: (
      <>
        Evaluate your federated learning models with built-in fairness metrics like
        <b> Demographic Parity (DP)</b> and <b>Equalized Odds (EO)</b>, calculated globally or per-client.
      </>
    ),
  },
  {
    title: 'Controlled Bias',
    Svg: require('@site/static/img/icon_bias.svg').default,
    description: (
      <>
        Systematically inject <b>Attribute Skew</b> (demographic imbalance) and
        <b> Value Skew</b> (label correlation bias) to test the robustness of your fair FL algorithms.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
