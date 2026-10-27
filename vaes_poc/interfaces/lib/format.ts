/** O score é um erro quadrático médio pequeno (~0,001): mostrado multiplicado por 1000. */
export const fmtScore = (score: number) =>
  (score * 1000).toLocaleString("pt-BR", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

export const byScore = <T extends { score: number }>(items: T[]) =>
  [...items].sort((a, b) => b.score - a.score);
